from text2sql.execution.sandbox import ExecutionStatus, SQLiteSandbox


def test_executes_read_only_query(sample_database):
    result = SQLiteSandbox().execute(sample_database, "SELECT name FROM customers ORDER BY id")
    assert result.status is ExecutionStatus.SUCCESS
    assert result.rows == (("Ada",), ("Lin",))


def test_blocks_writes(sample_database):
    result = SQLiteSandbox().execute(sample_database, "DELETE FROM customers")
    assert result.status is ExecutionStatus.FORBIDDEN
    verify = SQLiteSandbox().execute(sample_database, "SELECT COUNT(*) FROM customers")
    assert verify.rows == ((2,),)


def test_blocks_multiple_statements(sample_database):
    result = SQLiteSandbox().execute(sample_database, "SELECT 1; DELETE FROM customers")
    assert result.status is ExecutionStatus.ERROR


def test_interrupts_expensive_query(sample_database):
    sql = """
    WITH RECURSIVE counter(x) AS (
      SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < 100000000
    )
    SELECT SUM(x) FROM counter
    """
    result = SQLiteSandbox(timeout_seconds=0.001, progress_steps=1).execute(sample_database, sql)
    assert result.status is ExecutionStatus.TIMEOUT


def test_enforces_row_limit(sample_database):
    result = SQLiteSandbox(max_rows=1).execute(sample_database, "SELECT name FROM customers")
    assert result.status is ExecutionStatus.TOO_MANY_ROWS

