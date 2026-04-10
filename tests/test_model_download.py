"""
Slow integration tests — verify embedding models load and encode without HF_TOKEN.

Run only when explicitly requested:
    pytest -m slow tests/test_model_download.py -v

Skipped in CI (no model downloads there).
"""

import os
import pytest
import beacon.indexer.embedder as embedder_mod
from beacon.indexer.embedder import SentenceEncoder


@pytest.mark.slow
class TestModelDownload:
    """Model download tests — require network access and disk space."""

    def setup_method(self):
        """Clear HF_TOKEN and reset the global encoder singleton."""
        self._original_token = os.environ.pop("HF_TOKEN", None)
        os.environ.update({
            "TRANSFORMERS_VERBOSITY": "error",
            "HF_HUB_VERBOSITY": "error",
            "PYTHONWARNINGS": "ignore",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "LOGURU_LEVEL": "ERROR",
            "BEACON_QUIET": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "DISABLE_TQDM": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "HF_HUB_DISABLE_TQDM": "1",
        })
        # Reset singleton so each test starts with a fresh encoder
        embedder_mod._encoder = None

    def teardown_method(self):
        """Restore original HF_TOKEN and reset encoder singleton."""
        if self._original_token is not None:
            os.environ["HF_TOKEN"] = self._original_token
        else:
            os.environ.pop("HF_TOKEN", None)
        embedder_mod._encoder = None

    def test_default_model_loads_without_token(self):
        """Default embedding model (jina-embeddings-v2-base-code) loads and encodes."""
        encoder = embedder_mod.get_encoder()
        assert isinstance(encoder, SentenceEncoder)

        test_texts = ["def hello(): pass", "class Test:\n    pass"]
        embeddings = encoder.encode(test_texts)
        assert embeddings is not None, (
            f"Model failed to encode texts. Encoder error: {encoder.error!r}"
        )
        assert embeddings.shape[0] == len(test_texts)
        assert embeddings.shape[1] > 0

    def test_large_model_loads_without_token(self):
        """jina-code-embeddings-1.5b loads and encodes without HF_TOKEN."""
        encoder = SentenceEncoder("jinaai/jina-code-embeddings-1.5b")
        test_texts = ["def test(): return 1"]
        embeddings = encoder.encode(test_texts)
        assert embeddings is not None, (
            f"Large model failed to encode texts. Encoder error: {encoder.error!r}"
        )
        assert embeddings.shape[0] == len(test_texts)
        assert embeddings.shape[1] > 0
