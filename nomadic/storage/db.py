#!/usr/bin/env python3
"""Database connector, config, and schema helpers for tracked chat sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import pathlib
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from .. import paths

PACKAGE_ROOT = files("nomadic.storage")
SQLITE_SCHEMA_RESOURCE = PACKAGE_ROOT.joinpath("sql", "sqlite_schema.sql")
POSTGRES_SCHEMA_RESOURCE = PACKAGE_ROOT.joinpath("sql", "postgres_schema.sql")
TASKS_SCHEMA_RESOURCE = PACKAGE_ROOT.joinpath("sql", "tasks.sql")
DEFAULT_DB_PATH = paths.default_db_path()

SESSION_STATUSES = {"active", "paused", "closed", "archived"}
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
DATABASE_KINDS = {"sqlite3", "postgres"}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "i", "in", "into", "is", "it", "of", "on", "or", "the", "to",
    "update", "use", "with", "you", "your",
}

DBRow = dict[str, Any]


@dataclass(frozen=True)
class DatabaseSettings:
    kind: str = "sqlite3"
    sqlite_path: str = str(DEFAULT_DB_PATH)
    postgres_dsn: str | None = None


@dataclass
class DBConnection:
    kind: str
    raw: Any
    target: str
    placeholder: str

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        return self.raw.execute(sql, params)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_config_dir(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return paths.config_dir(home)


def default_config_path(home: str | pathlib.Path | None = None) -> pathlib.Path:
    return default_config_dir(home) / "config.jsonc"


def resolve_config_path(config_path: str | pathlib.Path | None = None) -> pathlib.Path:
    if config_path is None:
        return default_config_path()
    return pathlib.Path(config_path).expanduser()


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _maybe_json(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _cfg_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_config_snapshot(config: Any) -> dict[str, Any]:
    keys = [
        "model",
        "host",
        "port",
        "temperature",
        "top_p",
        "max_tokens",
        "max_skill_chars",
        "max_context_tokens",
        "context_recompress_to",
    ]
    return {key: _cfg_value(config, key) for key in keys}


def strip_jsonc_comments(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            chars.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            chars.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(len(text), i + 2)
            continue

        chars.append(ch)
        i += 1

    return "".join(chars)


def _settings_from_config_payload(payload: Mapping[str, Any] | None) -> DatabaseSettings | None:
    if not payload:
        return None
    database = payload.get("database")
    if not isinstance(database, Mapping):
        return None

    kind = str(database.get("kind") or "sqlite3")
    sqlite_cfg = database.get("sqlite3") if isinstance(database.get("sqlite3"), Mapping) else {}
    postgres_cfg = database.get("postgres") if isinstance(database.get("postgres"), Mapping) else {}

    return DatabaseSettings(
        kind=kind,
        sqlite_path=str(sqlite_cfg.get("path") or DEFAULT_DB_PATH),
        postgres_dsn=postgres_cfg.get("dsn"),
    )


def load_config_file(config_path: str | pathlib.Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8")
    stripped = strip_jsonc_comments(raw_text).strip()
    if not stripped:
        return {}
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def load_database_settings_from_config(
    config_path: str | pathlib.Path | None = None,
) -> DatabaseSettings | None:
    return _settings_from_config_payload(load_config_file(config_path))


def _validate_settings(settings: DatabaseSettings) -> DatabaseSettings:
    if settings.kind not in DATABASE_KINDS:
        raise ValueError(f"Unsupported database kind: {settings.kind}")
    if settings.kind == "sqlite3":
        sqlite_path = settings.sqlite_path or str(DEFAULT_DB_PATH)
        return DatabaseSettings(kind="sqlite3", sqlite_path=sqlite_path, postgres_dsn=settings.postgres_dsn)
    if not settings.postgres_dsn:
        raise ValueError("Postgres database kind requires a DSN. Use --postgres-dsn or config.jsonc.")
    return settings


def resolve_db_settings(
    *,
    db_kind: str | None = None,
    db_path: str | pathlib.Path | None = None,
    postgres_dsn: str | None = None,
    config_path: str | pathlib.Path | None = None,
    use_config: bool = True,
) -> DatabaseSettings:
    config_settings = load_database_settings_from_config(config_path) if use_config else None

    resolved_kind = db_kind or (config_settings.kind if config_settings else None) or "sqlite3"
    resolved_sqlite_path = (
        str(db_path)
        if db_path is not None
        else (config_settings.sqlite_path if config_settings else None) or str(DEFAULT_DB_PATH)
    )
    resolved_postgres_dsn = (
        postgres_dsn
        if postgres_dsn is not None
        else (config_settings.postgres_dsn if config_settings else None)
    )
    return _validate_settings(
        DatabaseSettings(
            kind=resolved_kind,
            sqlite_path=resolved_sqlite_path,
            postgres_dsn=resolved_postgres_dsn,
        )
    )


def resolve_db_settings_from_args(args: Any, *, use_config: bool = True) -> DatabaseSettings:
    return resolve_db_settings(
        db_kind=getattr(args, "db_kind", None),
        db_path=getattr(args, "db_path", None),
        postgres_dsn=getattr(args, "postgres_dsn", None),
        config_path=getattr(args, "config_path", None),
        use_config=use_config,
    )


def _config_payload_from_settings(settings: DatabaseSettings) -> dict[str, Any]:
    return {
        "database": {
            "kind": settings.kind,
            "sqlite3": {
                "path": describe_database_target(
                    DatabaseSettings(kind="sqlite3", sqlite_path=settings.sqlite_path),
                    redact=False,
                ),
            },
            "postgres": {
                "dsn": settings.postgres_dsn or "",
            },
        }
    }


def render_config_jsonc(settings: DatabaseSettings) -> str:
    payload = _config_payload_from_settings(_validate_settings(settings))
    return (
        "// Nomadic runtime configuration.\n"
        "// This file is optional, but it becomes the default database source when present.\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )


def write_config_file(
    settings: DatabaseSettings,
    config_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    path = resolve_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_jsonc(settings), encoding="utf-8")
    return path


def redact_postgres_dsn(dsn: str) -> str:
    parts = urlsplit(dsn)
    if not parts.scheme:
        return dsn
    hostname = parts.hostname or ""
    credentials = ""
    if parts.username:
        credentials = parts.username
        if parts.password:
            credentials += ":***"
        credentials += "@"
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{credentials}{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def describe_database_target(settings: DatabaseSettings, *, redact: bool = True) -> str:
    settings = _validate_settings(settings)
    if settings.kind == "sqlite3":
        raw_path = str(settings.sqlite_path or DEFAULT_DB_PATH)
        if raw_path == ":memory:":
            return raw_path
        return str(pathlib.Path(raw_path).expanduser().resolve())
    dsn = settings.postgres_dsn or ""
    return redact_postgres_dsn(dsn) if redact else dsn


def add_db_selection_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--db-kind",
        default=default,
        choices=sorted(DATABASE_KINDS),
        dest="db_kind",
        help="Database backend to use for tracked kernel sessions",
    )
    parser.add_argument(
        "--db-path",
        default=default,
        dest="db_path",
        help="SQLite database path for tracked kernel sessions",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=default,
        dest="postgres_dsn",
        help="Postgres DSN for tracked kernel sessions",
    )
    parser.add_argument(
        "--config-path",
        default=default,
        dest="config_path",
        help="Path to config.jsonc (default: ~/.config/nomadic/config.jsonc)",
    )


def _content_json_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image_url":
                parts.append("[image attached]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _estimate_text_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _estimate_content_tokens(content_text: str | None, content_json: Any) -> int:
    total = 0
    if content_json is not None:
        if isinstance(content_json, list):
            for item in content_json:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    total += _estimate_text_tokens(item.get("text", ""))
                elif item.get("type") == "image_url":
                    total += 700
                else:
                    total += 32
        else:
            total += _estimate_text_tokens(str(content_json))
    elif content_text:
        total += _estimate_text_tokens(content_text)
    return max(1, total)


def _normalize_row(row: Any, cursor: Any) -> DBRow | None:
    if row is None:
        return None
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "keys"):
        keys = list(row.keys())
        return {key: row[key] for key in keys}
    columns = [column[0] for column in getattr(cursor, "description", []) or []]
    if isinstance(row, (tuple, list)) and columns:
        return dict(zip(columns, row))
    if isinstance(row, (tuple, list)):
        return {str(index): value for index, value in enumerate(row)}
    raise TypeError(f"Unsupported database row type: {type(row)!r}")


def _extract_scalar(row: Any, cursor: Any) -> Any:
    if row is None:
        raise RuntimeError("Expected a returned row but received none.")
    if isinstance(row, Mapping):
        return next(iter(row.values()))
    if hasattr(row, "keys"):
        keys = list(row.keys())
        return row[keys[0]]
    if isinstance(row, (tuple, list)):
        return row[0]
    return row


def _query_one(conn: DBConnection, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> DBRow | None:
    cursor = conn.execute(sql, params)
    return _normalize_row(cursor.fetchone(), cursor)


def _query_all(conn: DBConnection, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[DBRow]:
    cursor = conn.execute(sql, params)
    return [_normalize_row(row, cursor) for row in cursor.fetchall() if row is not None]


def _split_sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    chars: list[str] = []
    in_single = False
    in_double = False
    escaped = False

    for ch in script:
        if ch == "'" and not in_double and not escaped:
            in_single = not in_single
        elif ch == '"' and not in_single and not escaped:
            in_double = not in_double

        if ch == ";" and not in_single and not in_double:
            statement = "".join(chars).strip()
            if statement:
                statements.append(statement)
            chars = []
            escaped = False
            continue

        chars.append(ch)
        escaped = (ch == "\\") and not escaped
        if ch != "\\":
            escaped = False

    tail = "".join(chars).strip()
    if tail:
        statements.append(tail)
    return statements


def _execute_script(conn: DBConnection, script: str) -> None:
    if conn.kind == "sqlite3":
        conn.raw.executescript(script)
        return
    for statement in _split_sql_statements(script):
        conn.raw.execute(statement)


@contextmanager
def _transaction(conn: DBConnection) -> Iterator[None]:
    try:
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def _coerce_settings(value: DBConnection | DatabaseSettings | str | pathlib.Path | None) -> DatabaseSettings:
    if isinstance(value, DBConnection):
        return DatabaseSettings(
            kind=value.kind,
            sqlite_path=value.target if value.kind == "sqlite3" else str(DEFAULT_DB_PATH),
            postgres_dsn=value.target if value.kind == "postgres" else None,
        )
    if isinstance(value, DatabaseSettings):
        return _validate_settings(value)
    if isinstance(value, pathlib.Path):
        return DatabaseSettings(kind="sqlite3", sqlite_path=str(value))
    if isinstance(value, str):
        if value.startswith("postgresql://") or value.startswith("postgres://"):
            return DatabaseSettings(kind="postgres", postgres_dsn=value)
        return DatabaseSettings(kind="sqlite3", sqlite_path=value)
    return DatabaseSettings()


def connect(
    db: DBConnection | DatabaseSettings | str | pathlib.Path | None = None,
    *,
    db_kind: str | None = None,
    db_path: str | pathlib.Path | None = None,
    postgres_dsn: str | None = None,
    config_path: str | pathlib.Path | None = None,
    use_config: bool = True,
) -> DBConnection:
    if isinstance(db, DBConnection):
        return db

    if db_kind is not None or db_path is not None or postgres_dsn is not None or config_path is not None:
        settings = resolve_db_settings(
            db_kind=db_kind,
            db_path=db_path,
            postgres_dsn=postgres_dsn,
            config_path=config_path,
            use_config=use_config,
        )
    else:
        settings = _validate_settings(_coerce_settings(db))

    if settings.kind == "sqlite3":
        path_str = settings.sqlite_path or str(DEFAULT_DB_PATH)
        if path_str != ":memory:":
            path = pathlib.Path(path_str).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path_str = str(path)
            target = str(path.resolve())
        else:
            target = path_str
        raw = sqlite3.connect(path_str, timeout=30)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute("PRAGMA journal_mode = WAL")
        raw.execute("PRAGMA synchronous = NORMAL")
        raw.execute("PRAGMA busy_timeout = 5000")
        return DBConnection(kind="sqlite3", raw=raw, target=target, placeholder="?")

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Postgres support requires psycopg. Install project dependencies to use --db-kind postgres."
        ) from exc

    raw = psycopg.connect(settings.postgres_dsn)
    if hasattr(raw, "autocommit"):
        raw.autocommit = False
    return DBConnection(
        kind="postgres",
        raw=raw,
        target=describe_database_target(settings, redact=True),
        placeholder="%s",
    )


def load_schema(kind: str = "sqlite3") -> str:
    if kind == "postgres":
        return POSTGRES_SCHEMA_RESOURCE.read_text(encoding="utf-8")
    return SQLITE_SCHEMA_RESOURCE.read_text(encoding="utf-8")


def init_db(
    conn_or_path: DBConnection | DatabaseSettings | str | pathlib.Path = DEFAULT_DB_PATH,
) -> DBConnection:
    """Create the schema if needed and return the live connection."""
    conn = connect(conn_or_path) if not isinstance(conn_or_path, DBConnection) else conn_or_path
    with _transaction(conn):
        _execute_script(conn, load_schema(conn.kind))
    init_tasks_schema(conn)
    return conn


def provisional_chat_name() -> str:
    return f"Chat {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"


def suggest_chat_name(text: str, max_words: int = 6, max_chars: int = 48) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text or "")
    meaningful = [word for word in words if word.lower() not in _STOPWORDS]
    pick = meaningful[:max_words] or words[:max_words]
    if not pick:
        return provisional_chat_name()
    title = " ".join(pick).strip(" -_:;,.\n\t")
    if len(title) > max_chars:
        title = title[:max_chars].rstrip()
    return f"Chat: {title}" if title else provisional_chat_name()


def get_session(conn: DBConnection, session_id: str) -> DBRow | None:
    return _query_one(
        conn,
        f"SELECT * FROM chat_sessions WHERE id = {conn.placeholder}",
        (session_id,),
    )


def create_session(
    conn: DBConnection,
    *,
    session_id: str | None = None,
    chat_name: str | None = None,
    system_prompt: str,
    config: Any = None,
    status: str = "active",
    summary: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> DBRow:
    if status not in SESSION_STATUSES:
        raise ValueError(f"Invalid session status: {status}")

    session_id = session_id or str(uuid.uuid4())
    created_at = created_at or utc_now()
    chat_name = chat_name or provisional_chat_name()
    snapshot = _extract_config_snapshot(config)

    fields = {
        "id": session_id,
        "chat_name": chat_name,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "started_at": None,
        "ended_at": None,
        "system_prompt": system_prompt,
        "summary": summary,
        "metadata_json": _json(metadata),
    }
    fields.update(snapshot)

    columns = ", ".join(fields.keys())
    placeholders = ", ".join([conn.placeholder] * len(fields))
    with _transaction(conn):
        conn.execute(
            f"INSERT INTO chat_sessions ({columns}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
    session = get_session(conn, session_id)
    if session is None:
        raise RuntimeError(f"Session insert did not persist: {session_id}")
    return session


def resume_session(conn: DBConnection, session_id: str) -> DBRow:
    session = get_session(conn, session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    with _transaction(conn):
        conn.execute(
            f"""
            UPDATE chat_sessions
            SET status = 'active', ended_at = NULL, updated_at = {conn.placeholder}
            WHERE id = {conn.placeholder}
            """,
            (utc_now(), session_id),
        )
    session = get_session(conn, session_id)
    if session is None:
        raise RuntimeError(f"Session disappeared after resume: {session_id}")
    return session


def update_session(conn: DBConnection, session_id: str, **fields: Any) -> DBRow:
    allowed = {
        "chat_name",
        "status",
        "started_at",
        "ended_at",
        "model",
        "host",
        "port",
        "temperature",
        "top_p",
        "max_tokens",
        "max_skill_chars",
        "max_context_tokens",
        "context_recompress_to",
        "system_prompt",
        "summary",
        "message_count",
        "user_turn_count",
        "assistant_turn_count",
        "tool_event_count",
        "last_message_at",
        "metadata_json",
    }
    bad = sorted(set(fields) - allowed)
    if bad:
        raise KeyError(f"Unsupported session fields: {', '.join(bad)}")
    if not fields:
        session = get_session(conn, session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session
    if "status" in fields and fields["status"] not in SESSION_STATUSES:
        raise ValueError(f"Invalid session status: {fields['status']}")

    assignments = [f"{key} = {conn.placeholder}" for key in fields]
    values = list(fields.values())
    values.extend([utc_now(), session_id])
    with _transaction(conn):
        conn.execute(
            f"UPDATE chat_sessions SET {', '.join(assignments)}, updated_at = {conn.placeholder} "
            f"WHERE id = {conn.placeholder}",
            values,
        )
    session = get_session(conn, session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    return session


def _insert_with_auto_id(conn: DBConnection, sql: str, params: tuple[Any, ...] | list[Any]) -> int:
    if conn.kind == "postgres":
        cursor = conn.execute(f"{sql} RETURNING id", params)
        return int(_extract_scalar(cursor.fetchone(), cursor))
    cursor = conn.execute(sql, params)
    return int(cursor.lastrowid)


def append_message(
    conn: DBConnection,
    session_id: str,
    *,
    role: str,
    content_text: str | None = None,
    content_json: Any = None,
    content_type: str | None = None,
    raw_request_json: Any = None,
    raw_response_json: Any = None,
    token_estimate: int | None = None,
    parent_message_id: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> int:
    if role not in MESSAGE_ROLES:
        raise ValueError(f"Invalid message role: {role}")
    created_at = created_at or utc_now()
    content_type = content_type or (
        "multimodal" if isinstance(content_json, list) and len(content_json) > 1 else "text"
    )
    if content_text is None:
        content_text = _content_json_to_text(content_json)
    if token_estimate is None:
        token_estimate = _estimate_content_tokens(content_text, content_json)

    with _transaction(conn):
        next_index_row = _query_one(
            conn,
            f"SELECT COALESCE(MAX(message_index), 0) + 1 AS next_index "
            f"FROM chat_messages WHERE session_id = {conn.placeholder}",
            (session_id,),
        )
        next_index = int((next_index_row or {}).get("next_index", 1))
        message_id = _insert_with_auto_id(
            conn,
            f"""
            INSERT INTO chat_messages (
                session_id,
                message_index,
                role,
                content_type,
                content_text,
                content_json,
                raw_request_json,
                raw_response_json,
                token_estimate,
                created_at,
                parent_message_id,
                metadata_json
            ) VALUES (
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder}
            )
            """,
            (
                session_id,
                next_index,
                role,
                content_type,
                content_text,
                _json(content_json),
                _json(raw_request_json),
                _json(raw_response_json),
                token_estimate,
                created_at,
                parent_message_id,
                _json(metadata),
            ),
        )

        update_fragments = [
            "message_count = message_count + 1",
            f"last_message_at = {conn.placeholder}",
        ]
        update_values: list[Any] = [created_at]

        if role == "user":
            update_fragments.append("user_turn_count = user_turn_count + 1")
        elif role == "assistant":
            update_fragments.append("assistant_turn_count = assistant_turn_count + 1")
        elif role == "tool":
            update_fragments.append("tool_event_count = tool_event_count + 1")

        conn.execute(
            f"""
            UPDATE chat_sessions
            SET {', '.join(update_fragments)}, updated_at = {conn.placeholder}
            WHERE id = {conn.placeholder}
            """,
            update_values + [utc_now(), session_id],
        )

    return message_id


def record_attachment(
    conn: DBConnection,
    session_id: str,
    *,
    message_id: int | None,
    kind: str,
    file_path: str | pathlib.Path,
    file_name: str | None = None,
    mime_type: str | None = None,
    sha256: str | None = None,
    byte_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> int:
    path = pathlib.Path(file_path)
    file_name = file_name or path.name
    mime_type = mime_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if path.exists():
        byte_size = byte_size if byte_size is not None else path.stat().st_size
        if sha256 is None:
            digest = hashlib.sha256()
            digest.update(path.read_bytes())
            sha256 = digest.hexdigest()
    created_at = created_at or utc_now()

    with _transaction(conn):
        attachment_id = _insert_with_auto_id(
            conn,
            f"""
            INSERT INTO chat_attachments (
                session_id,
                message_id,
                kind,
                file_path,
                file_name,
                mime_type,
                sha256,
                byte_size,
                created_at,
                metadata_json
            ) VALUES (
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}
            )
            """,
            (
                session_id,
                message_id,
                kind,
                str(path),
                file_name,
                mime_type,
                sha256,
                byte_size,
                created_at,
                _json(metadata),
            ),
        )
    return attachment_id


def record_event(
    conn: DBConnection,
    session_id: str,
    *,
    event_type: str,
    event_name: str,
    status: str = "ok",
    payload: Mapping[str, Any] | None = None,
    message_id: int | None = None,
    created_at: str | None = None,
) -> int:
    created_at = created_at or utc_now()
    with _transaction(conn):
        event_id = _insert_with_auto_id(
            conn,
            f"""
            INSERT INTO chat_events (
                session_id,
                message_id,
                event_type,
                event_name,
                status,
                payload_json,
                created_at
            ) VALUES (
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}
            )
            """,
            (
                session_id,
                message_id,
                event_type,
                event_name,
                status,
                _json(payload),
                created_at,
            ),
        )
        conn.execute(
            f"UPDATE chat_sessions SET updated_at = {conn.placeholder} WHERE id = {conn.placeholder}",
            (created_at, session_id),
        )
    return event_id


def finalize_session_title(
    conn: DBConnection,
    session_id: str,
    *,
    user_text: str,
) -> DBRow:
    session = get_session(conn, session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    if session["started_at"]:
        return session

    started_at = utc_now()
    new_name = suggest_chat_name(user_text)
    with _transaction(conn):
        conn.execute(
            f"""
            UPDATE chat_sessions
            SET chat_name = {conn.placeholder}, started_at = {conn.placeholder}, updated_at = {conn.placeholder}
            WHERE id = {conn.placeholder}
            """,
            (new_name, started_at, started_at, session_id),
        )
    record_event(
        conn,
        session_id,
        event_type="session",
        event_name="title_generated",
        status="ok",
        payload={
            "chat_name": new_name,
            "source": "first_meaningful_user_turn",
            "seed_text": user_text[:160],
        },
        created_at=started_at,
    )
    session = get_session(conn, session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    return session


def close_session(
    conn: DBConnection,
    session_id: str,
    *,
    status: str = "closed",
    summary: str | None = None,
) -> DBRow:
    if status not in SESSION_STATUSES:
        raise ValueError(f"Invalid session status: {status}")
    ended_at = utc_now()
    updates: dict[str, Any] = {"status": status, "ended_at": ended_at}
    if summary is not None:
        updates["summary"] = summary
    session = update_session(conn, session_id, **updates)
    record_event(
        conn,
        session_id,
        event_type="session",
        event_name="closed",
        status="ok",
        payload={"status": status},
        created_at=ended_at,
    )
    return session


def load_messages(conn: DBConnection, session_id: str) -> list[DBRow]:
    return _query_all(
        conn,
        f"""
        SELECT *
        FROM chat_messages
        WHERE session_id = {conn.placeholder}
        ORDER BY message_index ASC, id ASC
        """,
        (session_id,),
    )


def build_runtime_history(
    conn: DBConnection,
    session_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    history = [{"role": "system", "content": session_row["system_prompt"]}]
    for row in load_messages(conn, session_row["id"]):
        role = row["role"]
        content = _maybe_json(row["content_json"]) if row["content_json"] is not None else row["content_text"]
        if role == "tool":
            role = "user"
        history.append({"role": role, "content": content})
    return history


def get_session_by_name(conn: DBConnection, name: str) -> DBRow | None:
    return _query_one(
        conn,
        f"SELECT * FROM chat_sessions WHERE chat_name = {conn.placeholder} ORDER BY updated_at DESC LIMIT 1",
        (name,),
    )


def list_sessions(
    conn: DBConnection,
    *,
    limit: int = 50,
    offset: int = 0,
    pwd: str | None = None,
) -> list[DBRow]:
    if pwd:
        # Simple string match in metadata_json for now, 
        # or more structured if using sqlite's json functions
        # For cross-db compatibility with minimal effort:
        return _query_all(
            conn,
            f"""
            SELECT *
            FROM chat_sessions
            WHERE metadata_json LIKE {conn.placeholder}
            ORDER BY updated_at DESC
            LIMIT {conn.placeholder} OFFSET {conn.placeholder}
            """,
            (f'%"pwd": "{pwd}"%', limit, offset),
        )

    return _query_all(
        conn,
        f"""
        SELECT *
        FROM chat_sessions
        ORDER BY updated_at DESC
        LIMIT {conn.placeholder} OFFSET {conn.placeholder}
        """,
        (limit, offset),
    )


# ---------------------------------------------------------------------------
# Task persistence helpers (FC-4)
# ---------------------------------------------------------------------------


def init_tasks_schema(conn: DBConnection) -> None:
    """Load and execute the tasks table DDL."""
    schema_sql = TASKS_SCHEMA_RESOURCE.read_text(encoding="utf-8")
    with _transaction(conn):
        _execute_script(conn, schema_sql)


def upsert_task(conn: DBConnection, task: Any) -> None:
    """Insert or update a task row from a :class:`nomadic.tasks.Task` instance."""
    d = task.to_dict()
    now = utc_now()
    artifacts_text = _json(d.get("artifacts", []))
    metadata_text = _json(d.get("metadata", {}))

    if conn.kind == "sqlite3":
        conn.execute(
            f"""
            INSERT OR REPLACE INTO tasks (
                id, objective, status, agent_id, session_id, parent_task_id,
                result, error, artifacts_json, metadata_json,
                created_at, updated_at
            ) VALUES (
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder}
            )
            """,
            (
                d["id"], d["objective"], d["status"], d.get("agent_id"),
                d.get("session_id"), d.get("parent_task_id"),
                d.get("result"), d.get("error"),
                artifacts_text, metadata_text,
                d.get("created_at", now), now,
            ),
        )
    else:
        conn.execute(
            f"""
            INSERT INTO tasks (
                id, objective, status, agent_id, session_id, parent_task_id,
                result, error, artifacts_json, metadata_json,
                created_at, updated_at
            ) VALUES (
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder},
                {conn.placeholder}, {conn.placeholder}, {conn.placeholder}, {conn.placeholder}
            )
            ON CONFLICT (id) DO UPDATE SET
                objective       = EXCLUDED.objective,
                status          = EXCLUDED.status,
                agent_id        = EXCLUDED.agent_id,
                session_id      = EXCLUDED.session_id,
                parent_task_id  = EXCLUDED.parent_task_id,
                result          = EXCLUDED.result,
                error           = EXCLUDED.error,
                artifacts_json  = EXCLUDED.artifacts_json,
                metadata_json   = EXCLUDED.metadata_json,
                updated_at      = EXCLUDED.updated_at
            """,
            (
                d["id"], d["objective"], d["status"], d.get("agent_id"),
                d.get("session_id"), d.get("parent_task_id"),
                d.get("result"), d.get("error"),
                artifacts_text, metadata_text,
                d.get("created_at", now), now,
            ),
        )
    conn.commit()


def get_task(conn: DBConnection, task_id: str) -> DBRow | None:
    """Fetch a single task row by id."""
    return _query_one(
        conn,
        f"SELECT * FROM tasks WHERE id = {conn.placeholder}",
        (task_id,),
    )


def list_tasks(
    conn: DBConnection,
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[DBRow]:
    """Return task rows with optional session/status filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if session_id is not None:
        clauses.append(f"session_id = {conn.placeholder}")
        params.append(session_id)
    if status is not None:
        clauses.append(f"status = {conn.placeholder}")
        params.append(status)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    return _query_all(
        conn,
        f"SELECT * FROM tasks{where} ORDER BY created_at DESC LIMIT {conn.placeholder}",
        params,
    )


def upsert_agent_state(conn: DBConnection, state: Mapping[str, Any]) -> None:
    """Insert or update an ``agent_states`` row from ``AgentInstance.to_state()``."""
    now = utc_now()
    ph = conn.placeholder
    conn.execute(
        f"""
        INSERT INTO agent_states (
            agent_id, task_id, session_id, status, depth, max_depth, system_prompt,
            history_json, current_tool_call_json, result, error, created_at, updated_at
        ) VALUES ({', '.join([ph] * 13)})
        ON CONFLICT (agent_id) DO UPDATE SET
            task_id                = excluded.task_id,
            session_id             = excluded.session_id,
            status                 = excluded.status,
            depth                  = excluded.depth,
            max_depth              = excluded.max_depth,
            system_prompt          = excluded.system_prompt,
            history_json           = excluded.history_json,
            current_tool_call_json = excluded.current_tool_call_json,
            result                 = excluded.result,
            error                  = excluded.error,
            updated_at             = excluded.updated_at
        """,
        (
            state["agent_id"], state.get("task_id"), state.get("session_id"),
            state.get("status", "idle"), int(state.get("depth", 0)),
            int(state.get("max_depth", 5)), state.get("system_prompt", ""),
            _json(state.get("history", [])) or "[]",
            _json(state["current_tool_call"]) if state.get("current_tool_call") else None,
            state.get("result"), state.get("error"), now, now,
        ),
    )
    conn.commit()


def _decode_agent_state(row: DBRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["history"] = _maybe_json(data.pop("history_json", None)) or []
    data["current_tool_call"] = _maybe_json(data.pop("current_tool_call_json", None))
    return data


def get_agent_state(conn: DBConnection, agent_id: str) -> dict[str, Any] | None:
    """Fetch one persisted agent state (history decoded), or ``None``."""
    return _decode_agent_state(
        _query_one(conn, f"SELECT * FROM agent_states WHERE agent_id = {conn.placeholder}", (agent_id,))
    )


def list_agent_states(
    conn: DBConnection,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return persisted agent states with optional filters, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("session_id", session_id), ("task_id", task_id), ("status", status)):
        if value is not None:
            clauses.append(f"{column} = {conn.placeholder}")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = _query_all(
        conn,
        f"SELECT * FROM agent_states{where} ORDER BY updated_at DESC LIMIT {conn.placeholder}",
        params,
    )
    return [_decode_agent_state(r) for r in rows if r is not None]  # type: ignore[misc]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nomadic-db", description="Session database helper")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create or upgrade the session schema")
    add_db_selection_args(init)

    ls = sub.add_parser("list-sessions", help="Print stored chat sessions")
    add_db_selection_args(ls)
    ls.add_argument("--limit", default=20, type=int, help="Maximum sessions to print")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        settings = resolve_db_settings_from_args(args, use_config=True)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "init":
        conn = connect(settings)
        try:
            init_db(conn)
            print(f"Initialized {settings.kind} DB at {describe_database_target(settings)}")
        finally:
            conn.close()
        return

    if args.command == "list-sessions":
        conn = connect(settings)
        try:
            init_db(conn)
            rows = list_sessions(conn, limit=args.limit)
            for row in rows:
                print(
                    f"{row['id']} | {row['chat_name']} | {row['status']} | "
                    f"{row['updated_at']} | messages={row['message_count']}"
                )
        finally:
            conn.close()
        return


if __name__ == "__main__":
    main()
