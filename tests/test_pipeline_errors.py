import unittest

from core.pipeline import RAGPipeline
from core.errors import EmbeddingError


class FakeLoader:
    def load(self, url=None, uploaded_file=None):
        return ["doc"]


class FakeRetriever:
    def create_index(self, split_docs, embeddings, vectorstore_path):
        # no-op
        self.created = True

    def load_index(self, embeddings, vectorstore_path):
        return "vs"

    def run_qa(self, llm, query):
        return {"answer": "ok"}


class TestPipelineErrors(unittest.TestCase):
    def test_index_requires_embeddings(self):
        pipeline = RAGPipeline(llm=None, embeddings=None, loader=FakeLoader(), retriever=FakeRetriever())
        with self.assertRaises(EmbeddingError):
            pipeline.index_documents(["chunk"])


if __name__ == "__main__":
    unittest.main()
