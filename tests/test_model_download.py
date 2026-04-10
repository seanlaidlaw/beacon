#!/usr/bin/env python3
"""
Unit test for model download without HF_TOKEN.
Verifies that beacon can download embedding models without authentication.
"""

import os
import sys
import unittest
from unittest.mock import patch

import pytest

from beacon.indexer.embedder import get_encoder, SentenceEncoder


@pytest.mark.slow
class TestModelDownload(unittest.TestCase):
    """Test model download without HF_TOKEN (skipped in CI — requires large model download)."""

    def setUp(self):
        """Clear HF_TOKEN from environment."""
        self.original_token = os.environ.get('HF_TOKEN')
        if 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']
        # Also set other env vars to suppress logging
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['HF_HUB_VERBOSITY'] = 'error'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['LOGURU_LEVEL'] = 'ERROR'
        os.environ['BEACON_QUIET'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['DISABLE_TQDM'] = '1'
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['HF_HUB_DISABLE_TQDM'] = '1'

    def tearDown(self):
        """Restore original HF_TOKEN."""
        if self.original_token is not None:
            os.environ['HF_TOKEN'] = self.original_token
        elif 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']

    def test_default_model_loads_without_token(self):
        """Test that default embedding model loads without HF_TOKEN."""
        try:
            encoder = get_encoder()
            # The encoder loads lazily, trigger load
            self.assertIsInstance(encoder, SentenceEncoder)
            # Try encoding a small text to ensure model works
            test_texts = ["def hello(): pass", "class Test:\n    pass"]
            # This will trigger actual model load
            embeddings = encoder.encode(test_texts)
            # If model fails to load, encode returns None
            self.assertIsNotNone(embeddings, "Model failed to encode texts")
            self.assertEqual(embeddings.shape[0], len(test_texts))
            print(f"Model loaded successfully, embedding dimension: {embeddings.shape[1]}")
        except Exception as e:
            self.fail(f"Model loading failed with error: {e}")

    def test_large_model_loads_without_token(self):
        """Test that jina-code-embeddings-1.5b loads without HF_TOKEN."""
        # Temporarily override config to use large model
        with patch('beacon.indexer.embedder._current_model') as mock_current:
            mock_current.return_value = 'jinaai/jina-code-embeddings-1.5b'
            try:
                encoder = get_encoder()
                self.assertIsInstance(encoder, SentenceEncoder)
                # Trigger load
                test_texts = ["def test(): return 1"]
                embeddings = encoder.encode(test_texts)
                self.assertIsNotNone(embeddings, "Large model failed to encode texts")
                print(f"Large model loaded successfully, embedding dimension: {embeddings.shape[1]}")
            except Exception as e:
                self.fail(f"Large model loading failed with error: {e}")


if __name__ == '__main__':
    unittest.main()