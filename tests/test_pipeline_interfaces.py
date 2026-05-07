import unittest

from core.pipeline import RAGPipeline


class FakeLoader:
    def __init__(self):
        self.called_with = None

    def load(self, url=None, uploaded_file=None):
        self.called_with = (url, uploaded_file)
        return ["doc1"]


class FakeRetriever:
    def __init__(self):
        self.create_index_called = False
        self.load_index_called = False
        self.run_qa_called_with = None
        self.vectorstore = "fake_vectorstore"

    def create_index(self, split_docs, embeddings, vectorstore_path):
        self.create_index_called = True
        self.created_docs = split_docs

    def load_index(self, embeddings, vectorstore_path):
        self.load_index_called = True
        return self.vectorstore

    def run_qa(self, llm, query):
        self.run_qa_called_with = (llm, query)
        return {"answer": "fake answer", "sources": "fake source"}


class TestRAGPipelineInterfaces(unittest.TestCase):
    def test_load_uses_loader(self):
        loader = FakeLoader()
        retriever = FakeRetriever()
        pipeline = RAGPipeline(llm=None, embeddings=object(), loader=loader, retriever=retriever)
        docs = pipeline.load_documents(url="http://example.com")
        self.assertEqual(docs, ["doc1"])
        self.assertEqual(loader.called_with[0], "http://example.com")

    def test_index_calls_retriever(self):
        loader = FakeLoader()
        retriever = FakeRetriever()
        pipeline = RAGPipeline(llm=None, embeddings=object(), loader=loader, retriever=retriever)
        pipeline.index_documents(["chunk1", "chunk2"])
        self.assertTrue(retriever.create_index_called)
        self.assertEqual(retriever.created_docs, ["chunk1", "chunk2"])

    def test_load_index_calls_retriever(self):
        loader = FakeLoader()
        retriever = FakeRetriever()
        pipeline = RAGPipeline(llm=None, embeddings=object(), loader=loader, retriever=retriever)
        vs = pipeline.load_index()
        self.assertTrue(retriever.load_index_called)
        self.assertEqual(vs, retriever.vectorstore)

    def test_query_calls_retriever_run_qa(self):
        loader = FakeLoader()
        retriever = FakeRetriever()
        pipeline = RAGPipeline(llm="llm-instance", embeddings=object(), loader=loader, retriever=retriever)
        result = pipeline.query("hello")
        self.assertEqual(result, {"answer": "fake answer", "sources": "fake source"})
        self.assertEqual(retriever.run_qa_called_with[1], "hello")


if __name__ == "__main__":
    unittest.main()
