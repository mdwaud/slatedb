#![allow(clippy::disallowed_types, clippy::disallowed_methods)]
//! Tests for `DbBuilder::with_wal_flush_sender`, which tails the database's durable writes.

use std::sync::Arc;
use std::time::Duration;

use slatedb::config::{FlushOptions, FlushType, Settings};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use slatedb::{Db, FlushedWal, WriteBatch};
use tokio::sync::mpsc;

/// No flush interval and a large WAL size limit, so only an explicit flush writes a WAL file
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
    tx: mpsc::Sender<Arc<FlushedWal>>,
) -> Db {
    Db::builder(path, object_store)
        .with_settings(manual_flush_settings())
        .with_wal_flush_sender(tx)
        .build()
        .await
        .unwrap()
}

/// Opens a db on a fresh in-memory store with a consumer channel of `capacity`
async fn open_tailing_db(path: &str, capacity: usize) -> (Db, mpsc::Receiver<Arc<FlushedWal>>) {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (tx, rx) = mpsc::channel(capacity);
    let db = open_db(path, object_store, tx).await;
    (db, rx)
}

/// Pins `FlushType::Wal` rather than calling `Db::flush`, because the WAL flush is under test
async fn flush_wal(db: &Db) -> Result<(), slatedb::Error> {
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::Wal,
    })
    .await
}

async fn recv(rx: &mut mpsc::Receiver<Arc<FlushedWal>>) -> Arc<FlushedWal> {
    tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("timed out waiting for a flushed wal")
        .expect("sender was dropped")
}

/// Drains the channel and reports whether it closed, rather than stalling
async fn drained_to_close(rx: &mut mpsc::Receiver<Arc<FlushedWal>>) -> bool {
    tokio::time::timeout(Duration::from_secs(10), async {
        while rx.recv().await.is_some() {}
    })
    .await
    .is_ok()
}

#[tokio::test]
async fn test_each_flush_sends_its_rows_in_seq_order() {
    // given: a write batch and a single write, each flushed into its own wal file
    let (db, mut rx) = open_tailing_db("/test_delivery", 16).await;
    let mut batch = WriteBatch::new();
    batch.put(b"k1", b"v1");
    batch.put(b"k2", b"v2");
    batch.delete(b"k3");
    db.write(batch).await.unwrap();
    flush_wal(&db).await.unwrap();
    db.put(b"k4", b"v4").await.unwrap();
    flush_wal(&db).await.unwrap();

    // then: the batch arrives whole, under one sequence number
    let first = recv(&mut rx).await;
    assert_eq!(first.rows.len(), 3);
    assert_eq!(first.first_seq, first.last_seq);
    for row in &first.rows {
        assert_eq!(row.seq, first.first_seq);
    }
    assert_eq!(first.rows[0].key.as_ref(), b"k1");
    assert_eq!(first.rows[1].key.as_ref(), b"k2");
    assert_eq!(first.rows[2].key.as_ref(), b"k3");
    assert!(first.rows[2].value.is_tombstone());

    // and: the next wal file follows it, with a contiguous sequence range
    let second = recv(&mut rx).await;
    assert_eq!(second.wal_id, first.wal_id + 1);
    // Sequence numbers only increase. They are not contiguous: an aborted transaction or an
    // explicit `WriteOptions::seqnum` consumes a sequence number without writing a row.
    assert!(second.first_seq > first.last_seq);
    assert_eq!(second.rows.len(), 1);
    assert_eq!(second.rows[0].key.as_ref(), b"k4");

    // and: closing drains what is left rather than discarding it
    db.put(b"k5", b"v5").await.unwrap();
    flush_wal(&db).await.unwrap();
    db.close().await.unwrap();
    let last = recv(&mut rx).await;
    assert_eq!(last.wal_id, second.wal_id + 1);
    assert!(rx.recv().await.is_none(), "close must close the channel");
}

#[tokio::test]
async fn test_replay_and_fencing_wals_send_nothing() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "/test_replay";

    // given: a wal file written by an earlier writer
    let (first_tx, _first_rx) = mpsc::channel(16);
    let db = open_db(path, object_store.clone(), first_tx).await;
    db.put(b"before", b"open").await.unwrap();
    flush_wal(&db).await.unwrap();
    db.close().await.unwrap();

    // when: the db reopens, replaying that file and writing a zero-byte fencing file
    let (tx, mut rx) = mpsc::channel(16);
    let db = open_db(path, object_store, tx).await;

    // then: neither one sends a message, though the replayed write is readable
    assert_eq!(
        db.get(b"before").await.unwrap().unwrap().as_ref(),
        b"open",
        "the write before the reopen should still be readable"
    );
    assert!(
        matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
        "replay and fencing wal files must not send a message, and must not close the channel"
    );

    // and: the first message is the first wal file this writer flushes
    db.put(b"after", b"open").await.unwrap();
    flush_wal(&db).await.unwrap();
    let message = recv(&mut rx).await;
    assert_eq!(message.rows.len(), 1);
    assert_eq!(message.rows[0].key.as_ref(), b"after");

    db.close().await.unwrap();
}

#[tokio::test]
async fn test_channel_closes_when_the_writer_is_fenced() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "/test_fenced_signal";

    // given: a writer that has flushed one wal file
    let (tx, mut rx) = mpsc::channel(16);
    let fenced = open_db(path, object_store.clone(), tx).await;
    fenced.put(b"a", b"1").await.unwrap();
    flush_wal(&fenced).await.unwrap();
    recv(&mut rx).await;

    // when: a second writer opens on the same path and fences the first
    let (tx, _rx) = mpsc::channel(16);
    let writer = open_db(path, object_store, tx).await;

    // then: the fenced writer cannot flush its next write. The write itself still succeeds,
    // because it only reaches the in-memory buffer.
    fenced.put(b"never", b"durable").await.unwrap();
    assert!(flush_wal(&fenced).await.is_err());

    // and: its consumer drains what arrived and then sees the channel close
    assert!(
        drained_to_close(&mut rx).await,
        "a fenced writer must close its consumer channel"
    );

    writer.close().await.unwrap();
}
