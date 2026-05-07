import unittest

from core.loaders import LangchainDocumentLoader
from core.errors import DocumentLoadError
from core.config import MAX_UPLOAD_BYTES


class FakeUploadedFile:
    def __init__(self, name: str, size: int):
        self.name = name
        self._data = b"x" * size

    def getbuffer(self):
        return self._data

    def read(self):
        return self._data


class TestLoaderErrors(unittest.TestCase):
    def test_oversized_upload_raises(self):
        loader = LangchainDocumentLoader()
        big = FakeUploadedFile("big.txt", MAX_UPLOAD_BYTES + 1)
        with self.assertRaises(DocumentLoadError):
            loader.load(uploaded_file=big)


if __name__ == "__main__":
    unittest.main()
