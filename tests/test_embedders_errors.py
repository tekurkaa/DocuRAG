import unittest

from core.embedders import WrapperEmbedder
from core.errors import EmbeddingError


class FakeEmbeddingsBad:
    def embed_documents(self, texts):
        # return fewer vectors than texts to simulate a broken embedder
        return [[0.1] for _ in range(max(0, len(texts) - 1))]


class TestEmbedderValidation(unittest.TestCase):
    def test_mismatched_vector_count_raises(self):
        fake = FakeEmbeddingsBad()
        wrapper = WrapperEmbedder(fake)
        with self.assertRaises(EmbeddingError):
            wrapper.embed_documents(["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
