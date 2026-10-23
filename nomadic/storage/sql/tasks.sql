-- Extension schema (SRS v0.3, FC-4 + section 6): persistent tasks and agent states
-- alongside the existing session model.

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

CREATE TABLE IF NOT EXISTS agent_states (
    agent_id                TEXT PRIMARY KEY,
    task_id                 TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    session_id              TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
    status                  TEXT NOT NULL DEFAULT 'idle',
    depth                   INTEGER NOT NULL DEFAULT 0,
    max_depth               INTEGER NOT NULL DEFAULT 5,
    system_prompt           TEXT NOT NULL DEFAULT '',
    history_json            TEXT NOT NULL DEFAULT '[]',
    current_tool_call_json  TEXT,
    result                  TEXT,
    error                   TEXT,
    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_states_task    ON agent_states(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_session ON agent_states(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_status  ON agent_states(status);
