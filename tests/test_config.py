"""Tests for beacon/config.py — user configuration I/O."""
import pytest
from pathlib import Path
from unittest.mock import patch

import beacon.config as config
from beacon.config import _read, _write, DEFAULT_MODEL_ID, MODELS


# ── _read / _write ────────────────────────────────────────────────────────────

class TestReadWrite:
    def test_round_trip(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        data = {"dense_model": "some/model", "other_key": "value"}
        _write(cfg_file, data)
        result = _read(cfg_file)
        assert result == data

    def test_empty_file_returns_empty_dict(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        assert _read(cfg_file) == {}

    def test_comments_are_skipped(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("# a comment\ndense_model: mymodel\n")
        result = _read(cfg_file)
        assert "dense_model" in result
        assert result["dense_model"] == "mymodel"
        assert not any(k.startswith("#") for k in result)

    def test_blank_lines_are_skipped(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("\n\ndense_model: mymodel\n\n")
        result = _read(cfg_file)
        assert list(result.keys()) == ["dense_model"]

    def test_lines_without_colon_are_skipped(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("no_colon_here\ndense_model: mymodel\n")
        result = _read(cfg_file)
        assert "no_colon_here" not in result
        assert "dense_model" in result

    def test_write_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "config.yaml"
        _write(nested, {"key": "val"})
        assert nested.exists()

    def test_values_with_colons_preserved(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        # value itself contains a colon — partition takes only first
        _write(cfg_file, {"url": "http://example.com"})
        result = _read(cfg_file)
        assert result["url"] == "http://example.com"

    def test_strips_whitespace_around_key_and_value(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("  dense_model  :  jinaai/model  \n")
        result = _read(cfg_file)
        assert result["dense_model"] == "jinaai/model"


# ── load / save ───────────────────────────────────────────────────────────────

class TestLoadSave:
    def test_load_returns_empty_when_file_missing(self, tmp_path):
        with patch.object(config, "_CONFIG_PATH", tmp_path / "nonexistent.yaml"):
            result = config.load()
        assert result == {}

    def test_load_returns_dict_from_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write(cfg_file, {"dense_model": "some/model"})
        with patch.object(config, "_CONFIG_PATH", cfg_file):
            result = config.load()
        assert result["dense_model"] == "some/model"

    def test_save_writes_to_config_path(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        with patch.object(config, "_CONFIG_PATH", cfg_file):
            config.save({"dense_model": "saved/model"})
        assert cfg_file.exists()
        result = _read(cfg_file)
        assert result["dense_model"] == "saved/model"

    def test_load_is_exception_safe(self, tmp_path):
        corrupt = tmp_path / "config.yaml"
        corrupt.write_bytes(b"\xff\xfe corrupt binary")
        # Should not raise, returns {}
        with patch.object(config, "_CONFIG_PATH", corrupt):
            result = config.load()
        assert isinstance(result, dict)


# ── get_dense_model / set_dense_model ─────────────────────────────────────────

class TestGetSetDenseModel:
    def test_default_model_when_no_config(self, tmp_path):
        with patch.object(config, "_CONFIG_PATH", tmp_path / "config.yaml"):
            result = config.get_dense_model()
        assert result == DEFAULT_MODEL_ID

    def test_set_then_get_returns_new_model(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        with patch.object(config, "_CONFIG_PATH", cfg_file):
            config.set_dense_model("jinaai/jina-code-embeddings-1.5b")
            result = config.get_dense_model()
        assert result == "jinaai/jina-code-embeddings-1.5b"

    def test_set_model_preserves_other_keys(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        _write(cfg_file, {"other_key": "other_value", "dense_model": "old/model"})
        with patch.object(config, "_CONFIG_PATH", cfg_file):
            config.set_dense_model("new/model")
            result = config.load()
        assert result["other_key"] == "other_value"
        assert result["dense_model"] == "new/model"

    def test_default_model_id_is_in_models_list(self):
        model_ids = [m["id"] for m in MODELS]
        assert DEFAULT_MODEL_ID in model_ids


# ── exists / config_path ──────────────────────────────────────────────────────

class TestExistsConfigPath:
    def test_exists_false_when_file_missing(self, tmp_path):
        with patch.object(config, "_CONFIG_PATH", tmp_path / "config.yaml"):
            assert config.exists() is False

    def test_exists_true_when_file_present(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("dense_model: x\n")
        with patch.object(config, "_CONFIG_PATH", cfg_file):
            assert config.exists() is True

    def test_config_path_returns_path_object(self):
        result = config.config_path()
        assert isinstance(result, Path)
