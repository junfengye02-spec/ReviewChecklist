import unittest

from tender_review.documents import ArtifactStore, InMemoryArtifactStore
from tender_review.documents.models import ArtifactWrite
from tender_review.retrieval.fakes import FakeEmbeddingProvider
from tender_review.retrieval.models import EmbeddingRequest
from tender_review.review import FakeLlmProvider, LlmProvider
from tender_review.review.models import LlmMessage, LlmRequest
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import CancelledError, NotFoundError


class OfflineFakeContractTests(unittest.TestCase):
    def setUp(self):
        self.call = CallContext(call_id="call-1", timeout_seconds=1, max_attempts=1)

    def test_memory_artifact_store_is_content_addressed_and_idempotent(self):
        store = InMemoryArtifactStore()
        artifact = ArtifactWrite(key="sha256/example", content=b"payload")

        first = store.put(artifact)
        second = store.put(artifact)

        self.assertIsInstance(store, ArtifactStore)
        self.assertEqual(first, second)
        self.assertTrue(first.key.startswith("sha256/"))
        self.assertEqual(store.get(first.key), b"payload")
        different = store.put(ArtifactWrite(key=first.key, content=b"different"))
        self.assertNotEqual(first.key, different.key)
        with self.assertRaises(NotFoundError):
            store.get("missing")

    def test_embedding_fake_is_deterministic_and_records_calls(self):
        provider = FakeEmbeddingProvider(dimensions=3)
        request = EmbeddingRequest(texts=("资格条件", "评标办法"), call=self.call)

        first = provider.embed(request)
        second = provider.embed(request)

        self.assertEqual(first.vectors, second.vectors)
        self.assertEqual(first.dimensions, 3)
        self.assertEqual(provider.calls, [request, request])

    def test_llm_fake_implements_protocol_without_network(self):
        provider = FakeLlmProvider(['{"result": true}'])
        request = LlmRequest(
            messages=(LlmMessage(role="user", content="review"),),
            call=self.call,
        )

        response = provider.complete(request)

        self.assertIsInstance(provider, LlmProvider)
        self.assertEqual(response.model, "fake-llm")
        self.assertEqual(response.content, '{"result": true}')
        self.assertEqual(provider.calls, [request])

    def test_external_fake_honors_cancelled_call_context(self):
        provider = FakeLlmProvider(["must-not-be-used"])
        request = LlmRequest(
            messages=(LlmMessage(role="user", content="review"),),
            call=CallContext(call_id="cancelled-call", cancelled=True),
        )

        with self.assertRaises(CancelledError):
            provider.complete(request)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
