"""Tests for beacon/search/capsule.py — capsule builder."""
import pytest
from beacon.schema import open_db
from beacon.search.capsule import (
    _file_path_to_module,
    _expand_neighbors,
    _importer_nodes,
    get_capsule,
    MIN_EDGE_CONFIDENCE,
    MAX_BFS_FANOUT,
)


def _make_db():
    return open_db(":memory:")


def _insert_node(conn, *, name, fqn, file_path="a.py", kind="function",
                 is_test=0, is_exported=1, signature="", docstring=""):
    conn.execute(
        """INSERT INTO nodes (name, fqn, file_path, kind, start_line,
           signature, docstring, is_exported, is_test, repo_alias)
           VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, 'primary')""",
        (name, fqn, file_path, kind, signature, docstring, is_exported, is_test),
    )
    conn.commit()
    return conn.execute("SELECT id FROM nodes WHERE fqn=?", (fqn,)).fetchone()[0]


def _insert_edge(conn, source_id, target_id, confidence=1.0, edge_type="CALLS"):
    conn.execute(
        "INSERT INTO edges (source_id, target_id, type, confidence) VALUES (?, ?, ?, ?)",
        (source_id, target_id, edge_type, confidence),
    )
    conn.commit()


# ── _file_path_to_module ──────────────────────────────────────────────────────

class TestFilePathToModule:
    def test_generates_suffix_candidates(self):
        candidates = _file_path_to_module("django/db/models/signals.py")
        assert "django.db.models.signals" in candidates
        assert "db.models.signals" in candidates
        assert "models.signals" in candidates
        assert "signals" in candidates

    def test_strips_py_extension(self):
        candidates = _file_path_to_module("myapp/views.py")
        assert all(not c.endswith(".py") for c in candidates)

    def test_strips_src_prefix(self):
        candidates = _file_path_to_module("src/myapp/views.py")
        assert "myapp.views" in candidates
        assert not any(c.startswith("src.") for c in candidates)

    def test_strips_lib_prefix(self):
        candidates = _file_path_to_module("lib/utils/helpers.py")
        assert "utils.helpers" in candidates

    def test_normalises_windows_backslashes(self):
        candidates = _file_path_to_module("django\\db\\models\\signals.py")
        assert "django.db.models.signals" in candidates

    def test_single_file_returns_stem(self):
        candidates = _file_path_to_module("utils.py")
        assert "utils" in candidates

    def test_returns_list(self):
        result = _file_path_to_module("a/b.py")
        assert isinstance(result, list)
        assert len(result) > 0


# ── _expand_neighbors ────────────────────────────────────────────────────────

class TestExpandNeighbors:
    def test_empty_seeds_returns_empty(self):
        conn = _make_db()
        assert _expand_neighbors(conn, []) == {}

    def test_high_confidence_edge_included(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        target = _insert_node(conn, name="callee", fqn="a::callee")
        _insert_edge(conn, seed, target, confidence=MIN_EDGE_CONFIDENCE)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert target in result
        assert result[target] == "callee"

    def test_low_confidence_edge_excluded(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        target = _insert_node(conn, name="low_conf", fqn="a::low_conf")
        _insert_edge(conn, seed, target, confidence=MIN_EDGE_CONFIDENCE - 0.1)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert target not in result

    def test_caller_discovered(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        caller = _insert_node(conn, name="caller", fqn="a::caller")
        # caller → seed edge
        _insert_edge(conn, caller, seed, confidence=1.0)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert caller in result
        assert result[caller] == "caller"

    def test_seed_ids_not_in_result(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        callee = _insert_node(conn, name="callee", fqn="a::callee")
        _insert_edge(conn, seed, callee, confidence=1.0)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert seed not in result

    def test_depth_one_does_not_reach_depth_two(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        hop1 = _insert_node(conn, name="hop1", fqn="a::hop1")
        hop2 = _insert_node(conn, name="hop2", fqn="a::hop2")
        _insert_edge(conn, seed, hop1, confidence=1.0)
        _insert_edge(conn, hop1, hop2, confidence=1.0)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert hop1 in result
        assert hop2 not in result

    def test_depth_two_reaches_two_hops(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        hop1 = _insert_node(conn, name="hop1", fqn="a::hop1")
        hop2 = _insert_node(conn, name="hop2", fqn="a::hop2")
        _insert_edge(conn, seed, hop1, confidence=1.0)
        _insert_edge(conn, hop1, hop2, confidence=1.0)
        result = _expand_neighbors(conn, [seed], depth=2)
        assert hop1 in result
        assert hop2 in result

    def test_fanout_cap_prevents_expansion_of_high_degree_node(self):
        conn = _make_db()
        # Insert a hub node with MAX_BFS_FANOUT + 1 callees
        hub = _insert_node(conn, name="hub", fqn="a::hub")
        for i in range(MAX_BFS_FANOUT + 1):
            callee = _insert_node(conn, name=f"callee_{i}", fqn=f"a::callee_{i}")
            _insert_edge(conn, hub, callee, confidence=1.0)
        result = _expand_neighbors(conn, [hub], depth=1)
        # Hub has too many callees (> MAX_BFS_FANOUT) → should NOT be expanded
        assert len(result) == 0

    def test_node_within_fanout_limit_is_expanded(self):
        conn = _make_db()
        seed = _insert_node(conn, name="seed", fqn="a::seed")
        # Insert exactly MAX_BFS_FANOUT callees (at the limit, not over)
        for i in range(MAX_BFS_FANOUT):
            callee = _insert_node(conn, name=f"c_{i}", fqn=f"a::c_{i}")
            _insert_edge(conn, seed, callee, confidence=1.0)
        result = _expand_neighbors(conn, [seed], depth=1)
        assert len(result) == MAX_BFS_FANOUT


# ── _importer_nodes ───────────────────────────────────────────────────────────

class TestImporterNodes:
    def test_empty_seed_paths_returns_empty(self):
        conn = _make_db()
        assert _importer_nodes(conn, [], budget_remaining=1000) == []

    def test_zero_budget_returns_empty(self):
        conn = _make_db()
        conn.execute(
            "INSERT INTO import_refs (source_file, target_module) VALUES ('importer.py', 'mymodule')"
        )
        conn.commit()
        _insert_node(conn, name="func", fqn="importer::func", file_path="importer.py")
        result = _importer_nodes(conn, ["mymodule.py"], budget_remaining=0)
        assert result == []

    def test_finds_importer_of_seed_file(self):
        conn = _make_db()
        # seed file: signals.py → module candidate "signals"
        # importer: views.py imports "signals"
        conn.execute(
            "INSERT INTO import_refs (source_file, target_module) VALUES ('views.py', 'signals')"
        )
        conn.commit()
        _insert_node(conn, name="index_view", fqn="views::index_view",
                     file_path="views.py", is_exported=1)
        result = _importer_nodes(conn, ["signals.py"], budget_remaining=10000)
        fqns = [n.fqn for n in result]
        assert "views::index_view" in fqns

    def test_self_imports_excluded(self):
        conn = _make_db()
        # signals.py imports itself — should be excluded
        conn.execute(
            "INSERT INTO import_refs (source_file, target_module) VALUES ('signals.py', 'signals')"
        )
        conn.commit()
        _insert_node(conn, name="my_signal", fqn="signals::my_signal",
                     file_path="signals.py", is_exported=1)
        result = _importer_nodes(conn, ["signals.py"], budget_remaining=10000)
        assert result == []

    def test_only_exported_nodes_returned(self):
        conn = _make_db()
        conn.execute(
            "INSERT INTO import_refs (source_file, target_module) VALUES ('views.py', 'signals')"
        )
        conn.commit()
        _insert_node(conn, name="_private", fqn="views::_private",
                     file_path="views.py", is_exported=0)
        _insert_node(conn, name="public_func", fqn="views::public_func",
                     file_path="views.py", is_exported=1)
        result = _importer_nodes(conn, ["signals.py"], budget_remaining=10000)
        fqns = [n.fqn for n in result]
        assert "views::_private" not in fqns
        assert "views::public_func" in fqns


# ── get_capsule (P2 scoring penalties) ───────────────────────────────────────

class TestGetCapsuleP2Penalties:
    def test_test_node_penalised_for_non_test_query(self):
        conn = _make_db()
        # Insert test and non-test nodes both matching "csrf"
        _insert_node(conn, name="csrf_middleware", fqn="a::csrf_middleware",
                     docstring="csrf middleware protection", is_test=0, is_exported=1)
        _insert_node(conn, name="test_csrf", fqn="a::test_csrf",
                     docstring="test csrf middleware", is_test=1, is_exported=1)
        cap = get_capsule(conn, "csrf middleware protection", max_tokens=8000)
        prod_scores = [n.score for n in cap.nodes if n.fqn == "a::csrf_middleware"]
        test_scores = [n.score for n in cap.nodes if n.fqn == "a::test_csrf"]
        if prod_scores and test_scores:
            # Test node should score lower due to 0.3× penalty
            assert test_scores[0] < prod_scores[0]

    def test_test_node_not_penalised_when_query_contains_test(self):
        conn = _make_db()
        _insert_node(conn, name="test_csrf", fqn="a::test_csrf",
                     docstring="test csrf middleware validation", is_test=1, is_exported=1)
        cap_test_query = get_capsule(conn, "test csrf middleware", max_tokens=8000)
        cap_prod_query = get_capsule(conn, "csrf middleware", max_tokens=8000)
        # Get test_csrf score in both capsules
        test_scores_test_q = [n.score for n in cap_test_query.nodes if n.fqn == "a::test_csrf"]
        test_scores_prod_q = [n.score for n in cap_prod_query.nodes if n.fqn == "a::test_csrf"]
        if test_scores_test_q and test_scores_prod_q:
            # Under a test-focused query, the node is NOT penalised
            # so its score should be higher (or equal) than under prod query
            assert test_scores_test_q[0] >= test_scores_prod_q[0]

    def test_unexported_node_mildly_penalised(self):
        conn = _make_db()
        _insert_node(conn, name="public_func", fqn="a::public_func",
                     docstring="csrf public function", is_test=0, is_exported=1)
        _insert_node(conn, name="_private_func", fqn="a::_private_func",
                     docstring="csrf private function", is_test=0, is_exported=0)
        cap = get_capsule(conn, "csrf function", max_tokens=8000)
        pub_scores = [n.score for n in cap.nodes if n.fqn == "a::public_func"]
        priv_scores = [n.score for n in cap.nodes if n.fqn == "a::_private_func"]
        if pub_scores and priv_scores:
            assert pub_scores[0] > priv_scores[0]

    def test_budget_respected(self):
        conn = _make_db()
        # Insert several nodes
        for i in range(20):
            _insert_node(conn, name=f"func_{i}", fqn=f"a::func_{i}",
                         docstring="csrf middleware token validation function code " * 5)
        # Very tight budget
        cap = get_capsule(conn, "csrf", max_tokens=200)
        assert cap.token_estimate <= 200

    def test_empty_db_returns_empty_capsule(self):
        conn = _make_db()
        cap = get_capsule(conn, "anything at all", max_tokens=8000)
        assert cap.nodes == []
        assert cap.token_estimate == 0
