-- tasks schema extension for SRS-ext-v1.1 (FC-4)
-- Adds persistent task/objective tracking alongside the existing session model.

CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    objective       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    agent_id        TEXT,
    session_id      TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
    parent_task_id  TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    result          TEXT,
    error           TEXT,
    artifacts_json  TEXT NOT NULL DEFAULT '[]',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_session    ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status     ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent     ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_agent      ON tasks(agent_id);
