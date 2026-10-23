from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
import uuid
from types import SimpleNamespace

from nomadic.storage import db


class DBConfigTests(unittest.TestCase):
    def test_strip_jsonc_comments_preserves_dsn_slashes(self):
        raw = """
        // Nomadic config
        {
          "database": {
            "kind": "postgres", // runtime backend
            "sqlite3": { "path": "kernel.sqlite3" },
            /* keep this DSN intact */
            "postgres": {
              "dsn": "postgresql://user:pass@localhost:5432/app?sslmode=disable"
            }
          }
        }
        """

        payload = json.loads(db.strip_jsonc_comments(raw))

        self.assertEqual(payload["database"]["kind"], "postgres")
        self.assertEqual(
            payload["database"]["postgres"]["dsn"],
            "postgresql://user:pass@localhost:5432/app?sslmode=disable",
        )

    def test_resolve_db_settings_prefers_cli_over_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = pathlib.Path(tmp) / "config.jsonc"
            config_path.write_text(
                """
                {
                  "database": {
                    "kind": "postgres",
                    "sqlite3": { "path": "/tmp/from-config.sqlite3" },
                    "postgres": { "dsn": "postgresql://cfg:secret@localhost:5432/config_db" }
                  }
                }
                """,
                encoding="utf-8",
            )

            configured = db.resolve_db_settings(config_path=config_path)
            overridden = db.resolve_db_settings(
                db_kind="sqlite3",
                db_path="override.sqlite3",
                config_path=config_path,
            )

        self.assertEqual(configured.kind, "postgres")
        self.assertEqual(
            configured.postgres_dsn,
            "postgresql://cfg:secret@localhost:5432/config_db",
        )
        self.assertEqual(overridden.kind, "sqlite3")
        self.assertEqual(overridden.sqlite_path, "override.sqlite3")

    def test_resolve_db_settings_from_args_uses_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = pathlib.Path(tmp) / "config.jsonc"
            config_path.write_text(
                """
                {
                  "database": {
                    "kind": "sqlite3",
                    "sqlite3": { "path": "/tmp/from-args.sqlite3" },
                    "postgres": { "dsn": "" }
                  }
                }
                """,
                encoding="utf-8",
            )
            args = SimpleNamespace(
                db_kind=None,
                db_path=None,
                postgres_dsn=None,
                config_path=str(config_path),
            )

            settings = db.resolve_db_settings_from_args(args, use_config=True)

        self.assertEqual(settings.kind, "sqlite3")
        self.assertEqual(settings.sqlite_path, "/tmp/from-args.sqlite3")


class SQLiteDatabaseTests(unittest.TestCase):
    def test_sqlite_init_and_session_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = pathlib.Path(tmp) / "kernel.sqlite3"
            settings = db.DatabaseSettings(kind="sqlite3", sqlite_path=str(db_path))
            conn = db.connect(settings)
            try:
                db.init_db(conn)
                session = db.create_session(conn, system_prompt="System prompt")
                message_id = db.append_message(
                    conn,
                    session["id"],
                    role="user",
                    content_text="hello from sqlite",
                )
                db.record_event(
                    conn,
                    session["id"],
                    event_type="session",
                    event_name="unit_test",
                    message_id=message_id,
                    payload={"source": "sqlite_test"},
                )

                sessions = db.list_sessions(conn, limit=10)
                messages = db.load_messages(conn, session["id"])
                history = db.build_runtime_history(conn, session)
            finally:
                conn.close()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], session["id"])
        self.assertEqual(messages[0]["content_text"], "hello from sqlite")
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[1]["role"], "user")
        self.assertEqual(db.describe_database_target(settings), str(db_path.resolve()))


class FakeCursor:
    def __init__(self, rows: list[object] | None = None, description: list[tuple[str, object]] | None = None):
        self._rows = list(rows or [])
        self.description = description or [("id", None)]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakePostgresRaw:
    def __init__(self):
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, sql: str, params=()):
        params_tuple = tuple(params)
        self.executed.append((sql, params_tuple))
        if "RETURNING id" in sql:
            return FakeCursor(rows=[{"id": 41}], description=[("id", None)])
        return FakeCursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class PostgresHelperTests(unittest.TestCase):
    def test_postgres_schema_uses_identity_columns(self):
        schema = db.load_schema("postgres")

        self.assertIn("GENERATED BY DEFAULT AS IDENTITY", schema)
        self.assertNotIn("AUTOINCREMENT", schema)

    def test_insert_with_auto_id_uses_returning_for_postgres(self):
        raw = FakePostgresRaw()
        conn = db.DBConnection(
            kind="postgres",
            raw=raw,
            target="postgresql://user@localhost:5432/app",
            placeholder="%s",
        )

        inserted_id = db._insert_with_auto_id(
            conn,
            "INSERT INTO chat_events (event_type) VALUES (%s)",
            ("session",),
        )

        self.assertEqual(inserted_id, 41)
        self.assertIn("RETURNING id", raw.executed[0][0])
        self.assertEqual(raw.executed[0][1], ("session",))

    @unittest.skipUnless(
        os.getenv("NOMADIC_TEST_POSTGRES_DSN"),
        "Set NOMADIC_TEST_POSTGRES_DSN to run live Postgres integration coverage.",
    )
    def test_live_postgres_roundtrip(self):
        settings = db.DatabaseSettings(
            kind="postgres",
            postgres_dsn=os.environ["NOMADIC_TEST_POSTGRES_DSN"],
        )
        conn = db.connect(settings)
        try:
            db.init_db(conn)
            session = db.create_session(
                conn,
                session_id=str(uuid.uuid4()),
                system_prompt="Integration system prompt",
            )
            rows = db.list_sessions(conn, limit=200)
        finally:
            conn.close()

        self.assertTrue(any(row["id"] == session["id"] for row in rows))


if __name__ == "__main__":
    unittest.main()
