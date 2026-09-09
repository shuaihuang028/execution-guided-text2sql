from text2sql.data.bird import BirdExample
from text2sql.data.prompt_builder import PromptBuilder
from text2sql.data.schema import (
    Column,
    DatabaseSchema,
    ForeignKey,
    TableSchema,
    attach_descriptions,
    extract_sqlite_schema,
    load_column_meanings_json,
    load_database_descriptions,
)
from text2sql.data.schema_pruning import prune_schema


def test_extracts_primary_and_foreign_keys(sample_database):
    schema = extract_sqlite_schema(sample_database, "shop")
    orders = next(table for table in schema.tables if table.name == "orders")
    assert next(column for column in orders.columns if column.name == "id").primary_key_position == 1
    assert orders.foreign_keys[0].target_table == "customers"
    assert orders.foreign_keys[0].source_column == "customer_id"


def test_prompt_is_sql_only_and_includes_evidence(sample_database):
    schema = extract_sqlite_schema(sample_database, "shop")
    example = BirdExample("shop:1", "shop", "Who spent the most?", "SELECT 1", "sum total")
    prompt = PromptBuilder().build(example, schema)
    assert "Return SQL only" in prompt
    assert 'FOREIGN KEY "orders"."customer_id" -> "customers"."id"' in prompt
    assert "Evidence:\nsum total" in prompt
    messages = PromptBuilder().build_messages(example, schema)
    assert [message["role"] for message in messages] == ["system", "user"]


def test_prompt_caps_long_descriptions():
    schema = DatabaseSchema(
        "shop", (TableSchema("orders", (Column("total", "REAL", True, description="abcdefgh"),)),)
    )
    rendered = PromptBuilder(max_description_chars=5).serialize_schema(schema)
    assert "abcde…" in rendered
    assert "abcdefgh" not in rendered


def test_prompt_handles_implicit_foreign_key_target():
    schema = DatabaseSchema(
        "shop",
        (
            TableSchema(
                "orders",
                (Column("customer_id", "INTEGER", False),),
                (ForeignKey("customer_id", "customers", None),),
            ),
        ),
    )
    assert '"customers".<PRIMARY KEY>' in PromptBuilder().serialize_schema(schema)


def test_loads_bird_column_descriptions(sample_database):
    root = sample_database.parent / "database_description"
    root.mkdir()
    (root / "orders.csv").write_text(
        "original_column_name,column_description\ntotal,Order value in dollars\n",
        encoding="utf-8",
    )
    schema = extract_sqlite_schema(sample_database, "shop")
    enriched = attach_descriptions(schema, load_database_descriptions(sample_database.parent))
    orders = next(table for table in enriched.tables if table.name == "orders")
    assert next(column for column in orders.columns if column.name == "total").description == (
        "Order value in dollars"
    )


def test_loads_windows_1252_descriptions(tmp_path):
    root = tmp_path / "database_description"
    root.mkdir()
    (root / "orders.csv").write_bytes(
        "original_column_name,column_description\ntotal,Price – net\n".encode("cp1252")
    )
    assert load_database_descriptions(tmp_path)["orders"]["total"] == "Price – net"


def test_loads_compact_column_meanings_json(tmp_path):
    path = tmp_path / "meanings.json"
    path.write_text(
        '{"shop|orders|total": "Order value", "other|x|y": "Ignore me"}',
        encoding="utf-8",
    )
    assert load_column_meanings_json(path, "shop") == {"orders": {"total": "Order value"}}


def test_pruning_keeps_relevant_table_and_fk_neighbor(sample_database):
    schema = extract_sqlite_schema(sample_database, "shop")
    pruned = prune_schema(schema, "customer order total", None, max_tables=2)
    assert {table.name for table in pruned.tables} == {"customers", "orders"}
