from text2sql.execution.result_match import compare_rows, query_has_order_by


def test_multiset_comparison_preserves_duplicates():
    assert compare_rows([(1,), (1,), (2,)], [(2,), (1,), (1,)], ordered=False).matches
    assert not compare_rows([(1,), (1,), (2,)], [(2,), (2,), (1,)], ordered=False).matches


def test_ordered_comparison_penalizes_reordering():
    assert not compare_rows([(1,), (2,)], [(2,), (1,)], ordered=True).matches


def test_numeric_tolerance_and_nulls():
    assert compare_rows([(1, None, 0.3)], [(1.0, None, 0.30000001)], ordered=True).matches
    assert not compare_rows([(None,)], [(0,)], ordered=True).matches


def test_order_by_detection_ignores_literals_and_comments():
    assert query_has_order_by("SELECT x FROM t ORDER BY x")
    assert not query_has_order_by("SELECT 'order by' FROM t -- ORDER BY x")

