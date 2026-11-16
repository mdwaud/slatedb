#![allow(clippy::disallowed_types, clippy::disallowed_methods)]
//! Tests for `DbBuilder::with_wal_flush_sender`, which lets an in-process consumer tail the
//! database's committed, durable writes.

use std::sync::Arc;
use std::time::Duration;

use slatedb::config::{FlushOptions, FlushType, Settings};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use slatedb::{Db, FlushedWal, WalReader, WriteBatch};
use tokio::sync::mpsc;

/// Settings that leave every WAL flush under the test's control. The flush interval is off and
/// the WAL size limit is large, so only an explicit flush writes a WAL file.
fn manual_flush_settings() -> Settings {
    Settings {
        flush_interval: None,
        l0_sst_size_bytes: 64 * 1024 * 1024,
        ..Settings::default()
    }
}

async fn open_db(
    path: &str,
    object_store: Arc<dyn ObjectStore>,
    settings: Settings,
    tx: mpsc::Sender<Arc<FlushedWal>>,
) -> Db {
    Db::builder(path, object_store)
        .with_settings(settings)
        .with_wal_flush_sender(tx)
        .build()
        .await
        .unwrap()
}

/// Opens a database on a fresh in-memory store with a consumer channel of `capacity`, and leaves
/// every WAL flush under the test's control.
async fn open_tailing_db(path: &str, capacity: usize) -> (Db, mpsc::Receiver<Arc<FlushedWal>>) {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (tx, rx) = mpsc::channel(capacity);
    let db = open_db(path, object_store, manual_flush_settings(), tx).await;
    (db, rx)
}

/// Flushes the WAL explicitly. The tests pin `FlushType::Wal` rather than calling `Db::flush`,
/// because the WAL flush is the behaviour under test.
async fn flush_wal(db: &Db) {
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::Wal,
    })
    .await
    .unwrap();
}

async fn recv(rx: &mut mpsc::Receiver<Arc<FlushedWal>>) -> Arc<FlushedWal> {
    tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("timed out waiting for a flushed wal")
        .expect("sender was dropped")
}

#[tokio::test]
async fn test_messages_arrive_in_wal_id_order_with_contiguous_seq_ranges() {
    let (db, mut rx) = open_tailing_db("/test_order", 16).await;

    for i in 0..4u8 {
        db.put(&[b'k', i], &[b'v', i]).await.unwrap();
        flush_wal(&db).await;
    }

    let mut previous: Option<Arc<FlushedWal>> = None;
    for i in 0..4u8 {
        let message = recv(&mut rx).await;
        assert_eq!(message.rows.len(), 1);
        assert_eq!(message.rows[0].key.as_ref(), &[b'k', i]);
        assert_eq!(message.first_seq, message.rows[0].seq);
        assert_eq!(message.last_seq, message.rows[0].seq);
        if let Some(previous) = &previous {
            assert!(
                message.wal_id > previous.wal_id,
                "wal id {} is not after {}",
                message.wal_id,
                previous.wal_id
            );
            assert_eq!(
                message.first_seq,
                previous.last_seq + 1,
                "seq range is not contiguous with the previous message"
            );
        }
        previous = Some(message);
    }

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_write_batch_rows_share_a_seq_and_arrive_in_one_message() {
    let (db, mut rx) = open_tailing_db("/test_batch", 16).await;

    let mut batch = WriteBatch::new();
    batch.put(b"k1", b"v1");
    batch.put(b"k2", b"v2");
    batch.delete(b"k3");
    db.write(batch).await.unwrap();
    flush_wal(&db).await;

    let message = recv(&mut rx).await;
    assert_eq!(message.rows.len(), 3);
    assert_eq!(message.first_seq, message.last_seq);
    for row in &message.rows {
        assert_eq!(row.seq, message.first_seq);
    }
    assert_eq!(message.rows[0].key.as_ref(), b"k1");
    assert_eq!(message.rows[1].key.as_ref(), b"k2");
    assert_eq!(message.rows[2].key.as_ref(), b"k3");
    assert!(message.rows[2].value.is_tombstone());

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_message_rows_match_wal_reader() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "/test_wal_reader_match";
    let (tx, mut rx) = mpsc::channel(16);
    let db = open_db(path, object_store.clone(), manual_flush_settings(), tx).await;

    for i in 0..3u8 {
        let mut batch = WriteBatch::new();
        batch.put([b'a', i], [b'v', i]);
        batch.put([b'b', i], [b'w', i]);
        db.write(batch).await.unwrap();
        flush_wal(&db).await;
    }

    let wal_reader = WalReader::new(path, object_store);
    for _ in 0..3 {
        let message = recv(&mut rx).await;
        let mut iterator = wal_reader.get(message.wal_id).iterator().await.unwrap();
        let mut rows = Vec::new();
        while let Some(row) = iterator.next().await.unwrap() {
            rows.push(row);
        }
        assert_eq!(rows, message.rows);
    }

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_full_channel_drops_messages_without_affecting_writes() {
    let (db, mut rx) = open_tailing_db("/test_full", 1).await;

    // Fill the channel and then overrun it. None of these writes may block or fail.
    for i in 0..6u8 {
        db.put(&[b'k', i], &[b'v', i]).await.unwrap();
        flush_wal(&db).await;
    }

    // Every write is still readable, so the drops cost only the messages.
    for i in 0..6u8 {
        let value = db.get(&[b'k', i]).await.unwrap().unwrap();
        assert_eq!(value.as_ref(), &[b'v', i]);
    }

    // The channel holds only the first message.
    let first = recv(&mut rx).await;
    assert_eq!(first.rows[0].key.as_ref(), &[b'k', 0]);

    // The next message shows the gap as a jump in wal_id.
    db.put(b"k6", b"v6").await.unwrap();
    flush_wal(&db).await;
    let next = recv(&mut rx).await;
    assert!(
        next.wal_id > first.wal_id + 1,
        "expected a wal id jump after the drops, got {} then {}",
        first.wal_id,
        next.wal_id
    );
    assert!(
        next.first_seq > first.last_seq + 1,
        "expected a seq gap after the drops"
    );

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_dropped_receiver_does_not_affect_writes() {
    let (db, rx) = open_tailing_db("/test_closed", 4).await;
    drop(rx);

    for i in 0..4u8 {
        db.put(&[b'k', i], &[b'v', i]).await.unwrap();
        flush_wal(&db).await;
    }
    for i in 0..4u8 {
        let value = db.get(&[b'k', i]).await.unwrap().unwrap();
        assert_eq!(value.as_ref(), &[b'v', i]);
    }

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_replay_and_fencing_wals_send_nothing() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "/test_replay";

    // Write a WAL file that the next open must replay, then leave without a clean close so the
    // data stays in the WAL.
    let (first_tx, _first_rx) = mpsc::channel(16);
    let db = open_db(
        path,
        object_store.clone(),
        manual_flush_settings(),
        first_tx,
    )
    .await;
    db.put(b"before", b"open").await.unwrap();
    flush_wal(&db).await;
    db.close().await.unwrap();

    // Reopen. The replayed WAL file and the zero-byte fencing WAL file send nothing.
    let (tx, mut rx) = mpsc::channel(16);
    let db = open_db(path, object_store, manual_flush_settings(), tx).await;
    assert_eq!(
        db.get(b"before").await.unwrap().unwrap().as_ref(),
        b"open",
        "the write before the reopen should still be readable"
    );
    assert!(
        rx.try_recv().is_err(),
        "replay and fencing wal files must not send a message"
    );

    // The first message is the first WAL file this writer flushes.
    db.put(b"after", b"open").await.unwrap();
    flush_wal(&db).await;
    let message = recv(&mut rx).await;
    assert_eq!(message.rows.len(), 1);
    assert_eq!(message.rows[0].key.as_ref(), b"after");
    assert_eq!(
        message.rows[0].value.as_bytes().expect("expected a value"),
        b"open".as_ref()
    );

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_messages_sent_for_interval_and_explicit_flushes() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let settings = Settings {
        flush_interval: Some(Duration::from_millis(10)),
        l0_sst_size_bytes: 64 * 1024 * 1024,
        ..Settings::default()
    };
    let (tx, mut rx) = mpsc::channel(16);
    let db = open_db("/test_triggers", object_store, settings, tx).await;

    // The flush interval flushes this write without an explicit call.
    db.put(b"interval", b"v").await.unwrap();
    let message = recv(&mut rx).await;
    assert_eq!(message.rows[0].key.as_ref(), b"interval");

    // An explicit WAL flush sends a message too.
    db.put(b"explicit", b"v").await.unwrap();
    flush_wal(&db).await;
    let message = recv(&mut rx).await;
    assert_eq!(message.rows[0].key.as_ref(), b"explicit");

    db.close().await.unwrap();
}
