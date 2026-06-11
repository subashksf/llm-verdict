"""Tests for DuckDB store initialization."""

from pathlib import Path

from llm_verdict.store.duck import init_db

TABLES_QUERY = (
    "SELECT table_name FROM information_schema.tables"
    " WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
)

VIEWS_QUERY = (
    "SELECT table_name FROM information_schema.tables"
    " WHERE table_schema = 'main' AND table_type = 'VIEW'"
)


def test_init_db_creates_tables(tmp_path: Path) -> None:
    """init_db creates all expected tables."""
    db_path = tmp_path / "test.duckdb"
    conn = init_db(db_path)

    tables = conn.execute(TABLES_QUERY).fetchall()
    table_names = {row[0] for row in tables}

    assert "runs" in table_names
    assert "trials" in table_names
    assert "scores" in table_names
    assert "verdicts" in table_names
    assert "annotations" in table_names
    conn.close()


def test_init_db_creates_views(tmp_path: Path) -> None:
    """init_db creates the convenience views."""
    db_path = tmp_path / "test.duckdb"
    conn = init_db(db_path)

    views = conn.execute(VIEWS_QUERY).fetchall()
    view_names = {row[0] for row in views}

    assert "model_summary" in view_names
    assert "category_summary" in view_names
    assert "head_to_head" in view_names
    assert "longitudinal" in view_names
    conn.close()


def test_init_db_idempotent(tmp_path: Path) -> None:
    """Calling init_db twice does not error."""
    db_path = tmp_path / "test.duckdb"
    conn1 = init_db(db_path)
    conn1.close()

    conn2 = init_db(db_path)
    tables = conn2.execute(TABLES_QUERY).fetchall()
    assert len(tables) == 5
    conn2.close()
