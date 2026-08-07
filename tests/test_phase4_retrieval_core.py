from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from tender_review.retrieval import (
    BM25Config,
    BM25Retriever,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    FusionRequest,
    FusionStrategy,
    HybridConfig,
    HybridRetriever,
    RRFConfig,
    RetrievalDocument,
    Retriever,
    RrfFusionStrategy,
    SearchHit,
    SearchRequest,
    SearchResult,
    VectorConfig,
    VectorRetriever,
    build_retriever,
    registered_retriever_kinds,
    tokenize,
)
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import CancelledError


CALL = CallContext(call_id="phase4-retrieval")


def document(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "document-1",
    section_path: tuple[str, ...] = (),
    page_start: int | None = None,
    page_end: int | None = None,
) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
    )


def request(
    query: str,
    *,
    limit: int = 10,
    document_ids: tuple[str, ...] = (),
    call: CallContext = CALL,
) -> SearchRequest:
    return SearchRequest(
        query=query,
        limit=limit,
        document_ids=document_ids,
        call=call,
    )


def result(name: str, *chunk_ids: str) -> SearchResult:
    return SearchResult(
        retriever=name,
        hits=tuple(
            SearchHit(
                chunk_id=chunk_id,
                document_id="document-1",
                text=f"text-{chunk_id}",
                score=float(len(chunk_ids) - index),
                source=name,
                rank=index + 1,
            )
            for index, chunk_id in enumerate(chunk_ids)
        ),
    )


class RetrievalContractTests(unittest.TestCase):
    def test_ports_are_runtime_checkable_for_all_core_implementations(self) -> None:
        provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"query": (1.0, 0.0)},
        )
        documents = (document("a", "alpha"), document("b", "beta"))
        bm25 = BM25Retriever(documents)
        vector = VectorRetriever(
            documents,
            provider,
            {"a": (1.0, 0.0), "b": (0.0, 1.0)},
        )
        fusion = RrfFusionStrategy()
        hybrid = HybridRetriever((bm25, vector), fusion)

        self.assertIsInstance(provider, EmbeddingProvider)
        self.assertIsInstance(bm25, Retriever)
        self.assertIsInstance(vector, Retriever)
        self.assertIsInstance(hybrid, Retriever)
        self.assertIsInstance(fusion, FusionStrategy)

    def test_blank_query_and_duplicate_document_filters_are_rejected(self) -> None:
        for query in ("", "   "):
            with self.subTest(query=query), self.assertRaises(ValidationError):
                request(query)
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            request("query", document_ids=("document-1", "document-1"))

    def test_ranked_result_rejects_duplicate_chunks_invalid_ranks_and_nan(self) -> None:
        valid = SearchHit(
            chunk_id="a",
            document_id="document-1",
            text="alpha",
            score=1.0,
            source="test",
            rank=1,
        )
        duplicate = valid.model_copy(update={"rank": 2})
        with self.assertRaisesRegex(ValidationError, "duplicate chunk_ids"):
            SearchResult(retriever="test", hits=(valid, duplicate))
        with self.assertRaisesRegex(ValidationError, "contiguous"):
            SearchResult(
                retriever="test",
                hits=(valid.model_copy(update={"rank": 2}),),
            )
        with self.assertRaisesRegex(ValidationError, "finite"):
            SearchHit(
                chunk_id="a",
                document_id="document-1",
                text="alpha",
                score=math.nan,
                source="test",
                rank=1,
            )

    def test_embedding_result_rejects_bad_dimensions_and_non_finite_values(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValidationError, "dimensions"):
            EmbeddingResult(
                model="test",
                dimensions=2,
                vectors=((1.0,),),
            )
        with self.assertRaisesRegex(ValidationError, "finite"):
            EmbeddingResult(
                model="test",
                dimensions=2,
                vectors=((math.inf, 0.0),),
            )


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            document(
                "bond-detailed",
                "投标保证金 保证金",
                section_path=("投标人须知", "保证金"),
                page_start=4,
                page_end=5,
            ),
            document("bond-brief", "投标保证金", page_start=8, page_end=8),
            document("schedule", "合同工期要求", document_id="document-2"),
        )
        self.retriever = BM25Retriever(
            self.documents,
            domain_terms=("投标保证金",),
        )

    def test_ranking_scores_top_k_filter_and_evidence_metadata(self) -> None:
        search = self.retriever.search(request("投标保证金", limit=1))

        self.assertEqual([hit.chunk_id for hit in search.hits], ["bond-detailed"])
        self.assertGreater(search.hits[0].score, 0)
        self.assertEqual(search.hits[0].rank, 1)
        self.assertEqual(search.hits[0].section_path, ("投标人须知", "保证金"))
        self.assertEqual((search.hits[0].page_start, search.hits[0].page_end), (4, 5))
        filtered = self.retriever.search(
            request("投标保证金", document_ids=("document-2",))
        )
        self.assertEqual(filtered.hits, ())

    def test_ties_use_chunk_id_and_queries_without_tokens_return_no_hits(self) -> None:
        tied = BM25Retriever(
            (document("b", "same term"), document("a", "same term"))
        ).search(request("same"))

        self.assertEqual([hit.chunk_id for hit in tied.hits], ["a", "b"])
        self.assertEqual(tied.hits[0].score, tied.hits[1].score)
        self.assertEqual(self.retriever.search(request("!!!")).hits, ())

    def test_duplicate_chunk_ids_and_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
            BM25Retriever((document("same", "one"), document("same", "two")))
        for kwargs in ({"k1": 0}, {"k1": math.inf}, {"b": -0.1}, {"b": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BM25Retriever(self.documents, **kwargs)

    def test_parsed_chunk_mapping_uses_raw_text_and_rejects_blank_ids(self) -> None:
        retriever = BM25Retriever(
            (
                {
                    "chunk_id": "parsed-1",
                    "document_id": "document-1",
                    "text": None,
                    "raw_text": "原始证据",
                    "section_path": ("原始章节",),
                    "page_start": 7,
                    "page_end": 7,
                },
            )
        )
        search = retriever.search(request("证据"))
        self.assertEqual(search.hits[0].text, "原始证据")
        self.assertEqual(search.hits[0].section_path, ("原始章节",))
        with self.assertRaisesRegex(TypeError, "chunk_id"):
            BM25Retriever(
                (
                    {
                        "chunk_id": " ",
                        "document_id": "document-1",
                        "raw_text": "text",
                    },
                )
            )

    def test_domain_terms_are_added_without_losing_deterministic_base_tokens(
        self,
    ) -> None:
        tokens = tokenize("投标保证金 ABC-123", domain_terms=("投标保证金",))

        self.assertEqual(tokens[:5], ("投", "标", "保", "证", "金"))
        self.assertIn("投标保证金", tokens)
        self.assertIn("abc", tokens)
        self.assertIn("123", tokens)

    def test_cancelled_calls_are_rejected_before_search(self) -> None:
        with self.assertRaises(CancelledError):
            self.retriever.search(
                request(
                    "投标保证金",
                    call=CallContext(call_id="cancelled", cancelled=True),
                )
            )


class VectorRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            document(
                "primary",
                "primary text",
                section_path=("资格条件",),
                page_start=2,
                page_end=2,
            ),
            document("near", "near text", page_start=9, page_end=9),
            document("opposite", "opposite text", document_id="document-2"),
        )
        self.provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={
                "primary text": (1.0, 0.0),
                "near text": (0.8, 0.2),
                "opposite text": (-1.0, 0.0),
                "eligibility": (1.0, 0.0),
            },
        )

    def test_offline_embedding_build_cosine_ranking_top_k_and_metadata(self) -> None:
        retriever = VectorRetriever.from_documents(
            self.documents,
            self.provider,
            call=CALL,
        )
        search = retriever.search(request("eligibility", limit=2))

        self.assertEqual([hit.chunk_id for hit in search.hits], ["primary", "near"])
        self.assertAlmostEqual(search.hits[0].score, 1.0)
        self.assertGreater(search.hits[1].score, 0.9)
        self.assertEqual(search.hits[0].section_path, ("资格条件",))
        self.assertEqual((search.hits[0].page_start, search.hits[0].page_end), (2, 2))
        self.assertEqual(
            self.provider.calls[0].texts, tuple(doc.text for doc in self.documents)
        )
        self.assertEqual(self.provider.calls[1].texts, ("eligibility",))

    def test_ties_filtering_and_negative_scores_are_deterministic(self) -> None:
        provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"query": (1.0, 0.0)},
        )
        retriever = VectorRetriever(
            self.documents,
            provider,
            {
                "primary": (1.0, 0.0),
                "near": (1.0, 0.0),
                "opposite": (-1.0, 0.0),
            },
        )

        search = retriever.search(request("query"))
        self.assertEqual(
            [hit.chunk_id for hit in search.hits],
            ["near", "primary", "opposite"],
        )
        self.assertEqual(search.hits[0].score, search.hits[1].score)
        self.assertEqual(search.hits[-1].score, -1.0)
        filtered = retriever.search(request("query", document_ids=("document-2",)))
        self.assertEqual([hit.chunk_id for hit in filtered.hits], ["opposite"])
        self.assertEqual(filtered.hits[0].rank, 1)

    def test_index_vector_shape_keys_values_and_norms_are_validated(self) -> None:
        cases = (
            ({"primary": (1.0, 0.0)}, "missing vectors"),
            (
                {
                    "primary": (1.0, 0.0),
                    "near": (1.0, 0.0),
                    "opposite": (1.0, 0.0),
                    "extra": (1.0, 0.0),
                },
                "unknown chunks",
            ),
            (
                {
                    "primary": (1.0, 0.0),
                    "near": (1.0,),
                    "opposite": (1.0, 0.0),
                },
                "same positive dimension",
            ),
            (
                {
                    "primary": (math.inf, 0.0),
                    "near": (1.0, 0.0),
                    "opposite": (1.0, 0.0),
                },
                "finite",
            ),
            (
                {
                    "primary": (0.0, 0.0),
                    "near": (1.0, 0.0),
                    "opposite": (1.0, 0.0),
                },
                "non-zero norms",
            ),
        )
        for vectors, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                VectorRetriever(self.documents, self.provider, vectors)

    def test_provider_count_dimension_return_type_and_zero_query_are_validated(
        self,
    ) -> None:
        class WrongCountProvider:
            def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
                del request
                return EmbeddingResult(
                    model="wrong-count",
                    dimensions=2,
                    vectors=((1.0, 0.0),),
                )

        with self.assertRaisesRegex(ValueError, "unexpected number"):
            VectorRetriever.from_documents(
                self.documents,
                WrongCountProvider(),
                call=CALL,
            )

        class WrongDimensionProvider:
            def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
                del request
                return EmbeddingResult(
                    model="wrong-dimension",
                    dimensions=3,
                    vectors=((1.0, 0.0, 0.0),),
                )

        retriever = VectorRetriever(
            self.documents,
            WrongDimensionProvider(),
            {
                "primary": (1.0, 0.0),
                "near": (1.0, 0.0),
                "opposite": (1.0, 0.0),
            },
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            retriever.search(request("query"))

        zero_provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"query": (0.0, 0.0)},
        )
        retriever = VectorRetriever(
            self.documents,
            zero_provider,
            {
                "primary": (1.0, 0.0),
                "near": (1.0, 0.0),
                "opposite": (1.0, 0.0),
            },
        )
        with self.assertRaisesRegex(ValueError, "non-zero norm"):
            retriever.search(request("query"))


class FusionAndHybridTests(unittest.TestCase):
    def test_rrf_merges_duplicate_chunks_and_breaks_equal_scores_by_chunk_id(
        self,
    ) -> None:
        fused = RrfFusionStrategy(k=60).fuse(
            FusionRequest(
                result_sets=(
                    result("bm25", "a", "b", "c"),
                    result("vector", "b", "a", "d"),
                ),
                limit=3,
            )
        )

        self.assertEqual([hit.chunk_id for hit in fused.hits], ["a", "b", "c"])
        self.assertEqual(fused.hits[0].score, fused.hits[1].score)
        self.assertEqual(fused.hits[0].source, "rrf:bm25+vector")
        self.assertEqual([hit.rank for hit in fused.hits], [1, 2, 3])
        self.assertAlmostEqual(fused.hits[0].score, 1 / 61 + 1 / 62)

    def test_rrf_rejects_conflicting_metadata_for_the_same_chunk(self) -> None:
        first = result("bm25", "same")
        conflicting = SearchResult(
            retriever="vector",
            hits=(
                first.hits[0].model_copy(
                    update={"document_id": "document-2", "source": "vector"}
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting metadata"):
            RrfFusionStrategy().fuse(
                FusionRequest(result_sets=(first, conflicting), limit=10)
            )

    def test_hybrid_returns_cross_section_evidence_through_retriever_contract(
        self,
    ) -> None:
        documents = (
            document(
                "qualification",
                "qualification bond requirement",
                section_path=("资格条件",),
                page_start=3,
                page_end=3,
            ),
            document(
                "payment",
                "payment terms",
                section_path=("合同条款",),
                page_start=20,
                page_end=21,
            ),
            document("noise", "unrelated appendix"),
        )
        provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"bond": (1.0, 0.0)},
        )
        hybrid = HybridRetriever(
            (
                BM25Retriever(documents),
                VectorRetriever(
                    documents,
                    provider,
                    {
                        "qualification": (1.0, 0.0),
                        "payment": (0.8, 0.2),
                        "noise": (0.0, 1.0),
                    },
                ),
            ),
            RrfFusionStrategy(),
            candidate_limit=3,
        )

        search = hybrid.search(request("bond", limit=2))

        self.assertEqual(search.retriever, "hybrid:rrf")
        self.assertEqual(
            [hit.chunk_id for hit in search.hits],
            ["qualification", "payment"],
        )
        self.assertEqual(
            [hit.section_path for hit in search.hits],
            [("资格条件",), ("合同条款",)],
        )
        self.assertEqual(search.hits[0].source, "rrf:bm25+vector")
        self.assertEqual(search.hits[1].source, "rrf:vector")

    def test_rrf_and_hybrid_configuration_boundaries(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(k=value), self.assertRaises(ValueError):
                RrfFusionStrategy(k=value)
        with self.assertRaisesRegex(ValueError, "at least two"):
            HybridRetriever((BM25Retriever(()),), RrfFusionStrategy())
        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            HybridRetriever(
                (BM25Retriever(()), BM25Retriever(())),
                RrfFusionStrategy(),
                candidate_limit=101,
            )


class RetrievalRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (document("a", "alpha"), document("b", "beta"))
        self.provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"query": (1.0, 0.0)},
        )
        self.vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0)}

    def test_registration_is_explicit_and_builds_all_three_variants(self) -> None:
        self.assertEqual(
            registered_retriever_kinds(),
            ("bm25", "vector", "hybrid"),
        )
        bm25 = build_retriever(BM25Config(), self.documents)
        vector = build_retriever(
            VectorConfig(),
            self.documents,
            embedding_provider=self.provider,
            vectors=self.vectors,
        )
        hybrid = build_retriever(
            HybridConfig(candidate_limit=2),
            self.documents,
            embedding_provider=self.provider,
            vectors=self.vectors,
        )

        self.assertIsInstance(bm25, BM25Retriever)
        self.assertIsInstance(vector, VectorRetriever)
        self.assertIsInstance(hybrid, HybridRetriever)
        self.assertEqual(hybrid.search(request("query", limit=1)).hits[0].chunk_id, "a")

    def test_registry_builds_vectors_from_embedding_when_not_precomputed(self) -> None:
        provider = FakeEmbeddingProvider(
            dimensions=2,
            vectors={"alpha": (1.0, 0.0), "beta": (0.0, 1.0)},
        )
        retriever = build_retriever(
            VectorConfig(),
            self.documents,
            embedding_provider=provider,
            index_call=CALL,
        )

        self.assertIsInstance(retriever, VectorRetriever)
        self.assertEqual(provider.calls[0].texts, ("alpha", "beta"))

    def test_invalid_configs_and_missing_vector_dependencies_fail_early(self) -> None:
        invalid_factories = (
            lambda: BM25Config(k1=0),
            lambda: BM25Config(b=1.1),
            lambda: BM25Config(domain_terms=("bond", " BOND ")),
            lambda: RRFConfig(k=0),
            lambda: HybridConfig(candidate_limit=101),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()
        with self.assertRaisesRegex(ValueError, "embedding_provider"):
            build_retriever(VectorConfig(), self.documents, vectors=self.vectors)
        with self.assertRaisesRegex(ValueError, "index_call"):
            build_retriever(
                VectorConfig(),
                self.documents,
                embedding_provider=self.provider,
            )
        with self.assertRaisesRegex(TypeError, "validated"):
            build_retriever("bm25", self.documents)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
