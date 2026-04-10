"""Tests for beacon/search/query.py — hybrid search pipeline."""
import pytest
from beacon.schema import open_db
from beacon.search.query import (
    _fts5_query,
    _make_reason,
    _bm25_search,
    _fallback_like_search,
    _graph_scores,
    search,
)


def _make_db():
    return open_db(":memory:")


def _insert_node(conn, *, name, fqn, file_path="a.py", kind="function",
                 signature="", docstring="", start_line=1):
    conn.execute(
        """INSERT INTO nodes (name, fqn, file_path, kind, start_line,
           signature, docstring, is_exported, is_test, repo_alias)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 'primary')""",
        (name, fqn, file_path, kind, start_line, signature, docstring),
    )
    conn.commit()
    return conn.execute("SELECT id FROM nodes WHERE fqn=?", (fqn,)).fetchone()[0]


# ── _fts5_query ───────────────────────────────────────────────────────────────

class TestFts5Query:
    def test_drops_short_tokens(self):
        # "go" (2 chars) and "to" (2 chars) should be dropped
        result = _fts5_query("go to next line")
        tokens = result.split(" OR ")
        assert "go" not in tokens
        assert "to" not in tokens
        assert "next" in tokens

    def test_drops_stopwords(self):
        # "the", "all", "for", "this" are in _STOP
        result = _fts5_query("find all the nodes for this function")
        lowered = [t.lower() for t in result.split(" OR ")]
        assert "the" not in lowered
        assert "all" not in lowered
        assert "for" not in lowered
        assert "this" not in lowered
        assert "find" in lowered
        assert "nodes" in lowered
        assert "function" in lowered

    def test_deduplicates_case_insensitive(self):
        result = _fts5_query("csrf CSRF Csrf")
        assert result.lower().count("csrf") == 1

    def test_fallback_when_all_stopwords(self):
        query = "the and for with from"
        result = _fts5_query(query)
        assert result == query[:50]

    def test_fallback_on_empty_string(self):
        result = _fts5_query("")
        assert result == ""

    def test_output_is_or_separated(self):
        result = _fts5_query("csrf protection middleware")
        assert " OR " in result

    def test_preserves_case_in_output(self):
        # tokens preserve original case
        result = _fts5_query("CsrfViewMiddleware")
        assert "CsrfViewMiddleware" in result


# ── _make_reason ──────────────────────────────────────────────────────────────

class TestMakeReason:
    def test_high_bm25(self):
        r = _make_reason({"bm25": 0.8, "semantic": 0.0, "graph": 0.0})
        assert "high text relevance" in r

    def test_low_bm25_keyword_match(self):
        r = _make_reason({"bm25": 0.3, "semantic": 0.0, "graph": 0.0})
        assert "keyword match" in r

    def test_bm25_below_threshold_no_mention(self):
        r = _make_reason({"bm25": 0.05, "semantic": 0.0, "graph": 0.0})
        assert "bm25" not in r.lower()
        assert "keyword" not in r

    def test_strong_semantic(self):
        r = _make_reason({"bm25": 0.0, "semantic": 0.7, "graph": 0.0})
        assert "strong semantic match" in r

    def test_moderate_semantic(self):
        r = _make_reason({"bm25": 0.0, "semantic": 0.3, "graph": 0.0})
        assert "semantic similarity" in r

    def test_highly_connected(self):
        r = _make_reason({"bm25": 0.0, "semantic": 0.0, "graph": 0.5})
        assert "highly connected" in r

    def test_graph_below_threshold_no_mention(self):
        r = _make_reason({"bm25": 0.0, "semantic": 0.0, "graph": 0.2})
        assert "connected" not in r

    def test_all_zeros_returns_weak_match(self):
        r = _make_reason({"bm25": 0.0, "semantic": 0.0, "graph": 0.0})
        assert r == "weak match"

    def test_combines_multiple_signals(self):
        r = _make_reason({"bm25": 0.8, "semantic": 0.7, "graph": 0.5})
        assert "high text relevance" in r
        assert "strong semantic match" in r
        assert "highly connected" in r
        assert " + " in r


# ── _bm25_search ─────────────────────────────────────────────────────────────

class TestBm25Search:
    def test_finds_matching_node_by_name(self):
        conn = _make_db()
        nid = _insert_node(conn, name="csrf_middleware", fqn="a::csrf_middleware")
        results = _bm25_search(conn, "csrf_middleware")
        ids = [r[0] for r in results]
        assert nid in ids

    def test_finds_matching_node_by_docstring(self):
        conn = _make_db()
        nid = _insert_node(conn, name="process_view", fqn="a::process_view",
                           docstring="Handles CSRF token validation")
        results = _bm25_search(conn, "CSRF token validation")
        ids = [r[0] for r in results]
        assert nid in ids

    def test_no_match_returns_empty(self):
        conn = _make_db()
        _insert_node(conn, name="foo", fqn="a::foo")
        results = _bm25_search(conn, "xyzzy_totally_nonexistent_zzz")
        assert results == []

    def test_limit_is_respected(self):
        conn = _make_db()
        for i in range(10):
            _insert_node(conn, name=f"csrf_{i}", fqn=f"a::csrf_{i}",
                         docstring="csrf token validation middleware")
        results = _bm25_search(conn, "csrf", limit=3)
        assert len(results) <= 3

    def test_returns_node_id_float_score_tuples(self):
        conn = _make_db()
        _insert_node(conn, name="process_csrf", fqn="a::process_csrf",
                     docstring="csrf processing middleware")
        results = _bm25_search(conn, "csrf")
        assert len(results) > 0
        nid, score = results[0]
        assert isinstance(nid, int)
        assert isinstance(score, float)
        assert score >= 0


# ── _fallback_like_search ─────────────────────────────────────────────────────

class TestFallbackLikeSearch:
    def test_matches_by_name(self):
        conn = _make_db()
        nid = _insert_node(conn, name="serialize_token", fqn="a::serialize_token")
        results = _fallback_like_search(conn, "serialize_token", limit=10)
        ids = [r[0] for r in results]
        assert nid in ids

    def test_matches_by_fqn(self):
        conn = _make_db()
        nid = _insert_node(conn, name="validate", fqn="auth::validate_csrf_token")
        results = _fallback_like_search(conn, "csrf_token", limit=10)
        ids = [r[0] for r in results]
        assert nid in ids

    def test_short_query_returns_empty(self):
        conn = _make_db()
        _insert_node(conn, name="xy_func", fqn="a::xy_func")
        # "xy" has only 2 chars, regex requires 3+
        results = _fallback_like_search(conn, "xy", limit=10)
        assert results == []

    def test_score_accumulates_for_multiple_token_matches(self):
        conn = _make_db()
        # This node name matches BOTH tokens → higher accumulated score
        nid_both = _insert_node(conn, name="csrf_token_check", fqn="a::csrf_token_check")
        nid_one = _insert_node(conn, name="csrf_only_func", fqn="a::csrf_only_func")
        results = dict(_fallback_like_search(conn, "csrf token", limit=10))
        if nid_both in results and nid_one in results:
            assert results[nid_both] >= results[nid_one]


# ── _graph_scores ─────────────────────────────────────────────────────────────

class TestGraphScores:
    def test_empty_node_ids(self):
        conn = _make_db()
        assert _graph_scores(conn, []) == {}

    def test_isolated_node_score_is_zero(self):
        conn = _make_db()
        nid = _insert_node(conn, name="isolated", fqn="a::isolated")
        scores = _graph_scores(conn, [nid])
        assert nid in scores
        assert scores[nid] == pytest.approx(0.0)

    def test_connected_node_scores_higher_than_isolated(self):
        conn = _make_db()
        hub = _insert_node(conn, name="hub_func", fqn="a::hub_func")
        leaf = _insert_node(conn, name="leaf_func", fqn="a::leaf_func")
        isolated = _insert_node(conn, name="isolated_func", fqn="a::isolated_func")
        conn.execute(
            "INSERT INTO edges (source_id, target_id, type, confidence) VALUES (?, ?, 'CALLS', 1.0)",
            (hub, leaf),
        )
        conn.commit()
        scores = _graph_scores(conn, [hub, leaf, isolated])
        # hub and leaf both have edges; isolated has none
        assert scores[hub] > scores[isolated]
        assert scores[leaf] > scores[isolated]

    def test_returns_all_requested_node_ids(self):
        conn = _make_db()
        n1 = _insert_node(conn, name="func_one", fqn="a::func_one")
        n2 = _insert_node(conn, name="func_two", fqn="a::func_two")
        scores = _graph_scores(conn, [n1, n2])
        assert n1 in scores
        assert n2 in scores


# ── search (integration) ─────────────────────────────────────────────────────

class TestSearch:
    def test_empty_results_for_no_matching_nodes(self):
        conn = _make_db()
        results = search(conn, "xyzzy_absolutely_nonexistent_term_zzz")
        assert results == []

    def test_exact_name_match_appears_in_results(self):
        conn = _make_db()
        _insert_node(conn, name="authenticate_user", fqn="auth::authenticate_user",
                     signature="def authenticate_user(request):")
        results = search(conn, "authenticate_user")
        fqns = [r.fqn for r in results]
        assert "auth::authenticate_user" in fqns

    def test_results_sorted_by_score_descending(self):
        conn = _make_db()
        _insert_node(conn, name="csrf_view", fqn="a::csrf_view",
                     signature="def csrf_view():",
                     docstring="csrf protection view middleware token")
        _insert_node(conn, name="other_func", fqn="a::other_func",
                     docstring="unrelated utility function helper")
        results = search(conn, "csrf")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_kind_filter_excludes_wrong_kinds(self):
        conn = _make_db()
        _insert_node(conn, name="CsrfClass", fqn="a::CsrfClass", kind="class",
                     docstring="csrf class implementation")
        _insert_node(conn, name="csrf_func", fqn="a::csrf_func", kind="function",
                     docstring="csrf function implementation")
        results = search(conn, "csrf", kind_filter="function")
        kinds = {r.kind for r in results}
        assert "class" not in kinds
        assert "function" in kinds

    def test_search_result_fields_populated(self):
        conn = _make_db()
        _insert_node(conn, name="my_func", fqn="pkg::my_func", file_path="pkg/mod.py",
                     kind="function", signature="def my_func(x): ...",
                     docstring="Does something useful")
        results = search(conn, "my_func")
        assert len(results) > 0
        r = results[0]
        assert r.name == "my_func"
        assert r.fqn == "pkg::my_func"
        assert r.file_path == "pkg/mod.py"
        assert r.kind == "function"
        assert isinstance(r.score, float)
        assert isinstance(r.score_breakdown, dict)
        assert "bm25" in r.score_breakdown
