import unittest
from unittest import mock

from ingestion.chunker import chunk_document
from retrieval.query import parse_retrieval_constraints


class LegalRetrievalTests(unittest.TestCase):
    def test_pipeline_modules_import(self):
        # Importing these modules catches broken signatures across the API,
        # ingestion, and retrieval layers without starting external services.
        import main  # noqa: F401
        import ingestion.ingest  # noqa: F401

    def test_extracts_explicit_article_constraint(self):
        constraints = parse_retrieval_constraints("What does Article 14 guarantee?")
        self.assertEqual(constraints.article_ref, "14")

    def test_preserves_lettered_article_references(self):
        constraints = parse_retrieval_constraints("What does Article 21A provide?")
        self.assertEqual(constraints.article_ref, "21A")

    def test_chunks_lettered_article_as_its_own_identifier(self):
        pages = [{"page": 1, "text": "21A. Right to education.—The State shall provide free education."}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(chunks[0]["article_ref"], "21A")

    def test_keeps_adjacent_articles_in_separate_chunks(self):
        pages = [{"page": 1, "text": (
            "14. Equality before law.—The State shall not deny to any person equality before the law "
            "or the equal protection of the laws within the territory of India.\n"
            "15. Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth.—"
            "The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, "
            "place of birth or any of them."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        article_14 = [chunk["text"] for chunk in chunks if chunk["article_ref"] == "14"]
        self.assertEqual(len(article_14), 1)
        self.assertIn("equal protection of the laws", article_14[0])
        self.assertNotIn("religion, race, caste", article_14[0])

    def test_accepts_pdf_replacement_character_as_heading_delimiter(self):
        pages = [{"page": 1, "text": (
            "14. Equality before law. �The State shall not deny equality.\n"
            "15. Prohibition of discrimination. �The State shall not discriminate."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(
            [chunk["article_ref"] for chunk in chunks if chunk["article_ref"]],
            ["14", "15"],
        )

    def test_skips_contents_and_accepts_wrapped_article_title(self):
        pages = [{"page": 1, "text": (
            "CONTENTS\n[2A. Sikkim to be associated with the Union.—Omitted.]\n"
            "WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India.\n"
            "14. Equality before law.—The State shall not deny equality.\n"
            "15. Prohibition of discrimination on grounds of religion, race, caste,\n"
            "sex or place of birth.—The State shall not discriminate."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(
            [chunk["article_ref"] for chunk in chunks if chunk["article_ref"]],
            ["14", "15"],
        )

    def test_does_not_treat_amendment_footnote_as_an_article(self):
        pages = [{"page": 1, "text": (
            "2. Subs. by the Constitution (Forty-second Amendment) Act.\n"
            "1. Name and territory of the Union.—India, that is Bharat, shall be a Union of States.\n"
            "2. Admission or establishment of new States.—Parliament may admit new States."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        article_1 = next(chunk["text"] for chunk in chunks if chunk["article_ref"] == "1")
        self.assertIn("Name and territory", article_1)
        self.assertNotIn("Subs. by", article_1)

    def test_extracts_article_wrapped_in_footnote_marker(self):
        pages = [{"page": 1, "text": (
            "1[150. Form of accounts of the Union and of the States.—"
            "The accounts of the Union and of the States shall be kept in such form.]"
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(chunks[0]["article_ref"], "150")

    def test_extracts_omitted_and_symbol_prefixed_articles(self):
        pages = [{"page": 1, "text": (
            "31. [Compulsory acquisition of property.].—Omitted.\n"
            "\uf02a\uf02a1[370. Temporary provisions with respect to the State.—"
            "Notwithstanding anything in this Constitution.]"
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(
            [chunk["article_ref"] for chunk in chunks if chunk["article_ref"]],
            ["31", "370"],
        )

    def test_footnote_cannot_absorb_next_article_heading(self):
        pages = [{"page": 1, "text": (
            "3. Lettered as sub-clause (d) by an amendment.\n"
            "281. Recommendations of the Finance Commission.—The President shall lay it before Parliament."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(
            [chunk["article_ref"] for chunk in chunks if chunk["article_ref"]],
            ["281"],
        )

    def test_extracts_article_with_internal_footnote_marker(self):
        pages = [{"page": 1, "text": (
            "334. 4[Reservation of seats and special representation to cease after a period].—"
            "The provisions of this Constitution shall cease."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(chunks[0]["article_ref"], "334")

    def test_extracts_omitted_article_without_a_dash(self):
        pages = [{"page": 1, "text": "[238. Omitted.]\n239. Administration of Union territories.—Text."}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        self.assertEqual(
            [chunk["article_ref"] for chunk in chunks if chunk["article_ref"]],
            ["238", "239"],
        )

    def test_does_not_tag_schedule_paragraphs_as_articles(self):
        pages = [{"page": 1, "text": (
            "WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India.\n"
            "14. Equality before law.—The State shall not deny equality.\n"
            "FIRST SCHEDULE\n[Articles 1 and 4]\n"
            "14. A schedule paragraph with its own numbering.—Schedule text."
        )}]
        chunks = chunk_document(pages, chunk_size_tokens=500, overlap_tokens=0)
        article_14 = [chunk["text"] for chunk in chunks if chunk["article_ref"] == "14"]
        self.assertEqual(len(article_14), 1)
        self.assertIn("Equality before law", article_14[0])
        self.assertTrue(any(chunk["article_ref"] is None and "schedule paragraph" in chunk["text"] for chunk in chunks))

    def test_rejects_invalid_chunk_overlap(self):
        with self.assertRaises(ValueError):
            chunk_document([{"page": 1, "text": "text"}], chunk_size_tokens=10, overlap_tokens=10)

    def test_overview_query_uses_document_overview_retrieval(self):
        from retrieval import hybrid

        with mock.patch.object(
            hybrid,
            "document_overview_search",
            return_value=[{"chunk_id": "1"}],
        ) as overview:
            result = hybrid.hybrid_search("What is this document about?", top_k=8)
        overview.assert_called_once_with(8, None, None)
        self.assertEqual(result, [{"chunk_id": "1"}])

    def test_overlap_preserves_complete_words(self):
        pages = [{"page": 1, "text": " ".join(f"word{i}" for i in range(160))}]
        chunks = chunk_document(pages, chunk_size_tokens=20, overlap_tokens=5)
        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertFalse(current["text"].startswith("[Article"))
            self.assertNotRegex(current["text"], r"\bwor$")

    def test_oversized_sentence_is_split_within_limit(self):
        text = " ".join(f"term{i}" for i in range(500))
        chunks = chunk_document(
            [{"page": 1, "text": text}],
            chunk_size_tokens=20,
            overlap_tokens=0,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk["text"].split()) < 120 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
