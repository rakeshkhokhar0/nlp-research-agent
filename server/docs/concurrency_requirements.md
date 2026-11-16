# Service-Layer Concurrency Requirements Specification

> **Note**: This document preserves the concurrency findings and transactional locking designs identified during the Database V1 Audit. These requirements are scheduled for implementation during the upcoming Service-Layer Development Phase.

---

## 1. Message Sequence Allocation
- **Database Constraint**: `UniqueConstraint("conversation_id", "sequence_number")` on `messages` table.
- **Concurrency Risk**: Concurrent messages arriving for the same conversation could calculate identical `MAX(sequence_number) + 1`, violating the unique constraint.
- **Service Requirement**:
  - Acquire a row-level lock on the parent `Conversation` within an explicit transaction:
    ```sql
    SELECT id FROM conversations WHERE id = :conversation_id FOR UPDATE;
    ```
  - Compute the next monotonic sequence number inside the lock:
    ```sql
    SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM messages WHERE conversation_id = :conversation_id;
    ```
  - Insert the message and update `conversations.last_message_at` in the same atomic transaction.

---

## 2. Chunk Version Activation
- **Database Constraint**: Partial unique index `ix_paper_chunks_paper_id_chunk_index_unique` on `(paper_id, chunk_index) WHERE is_current = TRUE`.
- **Concurrency Risk**: Multiple ingestion jobs for the same paper completing near-simultaneously could interleave their activation updates, resulting in unique constraint conflicts or inconsistent chunk states.
- **Service Requirement**:
  - Acquire a row-level lock on the parent `Paper` within an explicit transaction:
    ```sql
    SELECT id FROM papers WHERE id = :paper_id FOR UPDATE;
    ```
  - Validate that the ingestion job has `status == 'completed'` and has persisted chunks.
  - Check completion timestamps against currently active chunks to prevent older jobs that finish late from overwriting newer successfully completed versions.
  - Atomically update previous chunks (`is_current = FALSE`) and activate new chunks (`is_current = TRUE`).
  - Roll back cleanly if any step fails, leaving existing current chunks untouched.

---

## 3. Coordinated Paper File Deletion & Job Creation
- **Database Foreign Key**: `paper_files.id` referenced by `ingestion_jobs.paper_file_id` with `ON DELETE SET NULL`.
- **Concurrency Risk**: If file deletion only performs an in-memory check of active jobs, a concurrent request could create a new ingestion job for that file immediately after the check, leading to unexpected file deletion mid-processing.
- **Service Requirement**:
  - Both operations must acquire an exclusive row-level lock on the target `PaperFile`:
    ```sql
    SELECT id FROM paper_files WHERE id = :paper_file_id FOR UPDATE;
    ```
  - `create_ingestion_job`: Acquire lock on `PaperFile` before inserting the `IngestionJob`.
  - `delete_paper_file`: Acquire lock on `PaperFile`, verify that no active jobs (`pending` or `processing`) exist for this file inside the transaction, and only proceed with deletion if active job count is 0. Otherwise, raise `ActiveJobConflictError`.

