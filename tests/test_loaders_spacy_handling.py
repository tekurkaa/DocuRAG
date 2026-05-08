import unittest
import sys
import types

from core.loaders import LangchainDocumentLoader
from core.errors import DocumentLoadError


class FakeURLLoader:
    def __init__(self, urls=None):
        pass

    def load(self):
        # Simulate the permission error that spaCy's downloader can raise
        raise Exception(
            "Failed to install en_core_web_sm to /home/adminuser/venv/lib/python3.14/site-packages: "
            "[Errno 13] Permission denied: '/home/adminuser/venv/lib/python3.14/site-packages/en_core_web_sm'"
        )


class TestLangchainLoaderSpacyError(unittest.TestCase):
    def setUp(self):
        # Inject a fake langchain_community.document_loaders module to avoid
        # importing the real heavy dependency during unit tests.
        mod = types.ModuleType("langchain_community.document_loaders")
        mod.UnstructuredURLLoader = FakeURLLoader
        sys.modules["langchain_community"] = types.ModuleType("langchain_community")
        sys.modules["langchain_community.document_loaders"] = mod

    def tearDown(self):
        sys.modules.pop("langchain_community.document_loaders", None)
        sys.modules.pop("langchain_community", None)

    def test_spacy_install_permission_error_is_handled(self):
        loader = LangchainDocumentLoader()
        with self.assertRaises(DocumentLoadError) as cm:
            loader.load(url="http://example.com")
        err = str(cm.exception)
        self.assertIn("en_core_web_sm", err)
        self.assertIn("spaCy", err)


if __name__ == "__main__":
    unittest.main()
