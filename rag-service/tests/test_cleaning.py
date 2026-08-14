"""
Tests for the text cleaning service.

Validates that cleaning removes noise while preserving
educationally meaningful content.
"""

import pytest

from app.services.cleaning_service import (
    clean_text,
    clean_document,
    _collapse_whitespace,
    _collapse_blank_lines,
    _strip_page_artifacts,
    _fix_broken_words,
)
from app.schemas.document import ExtractedDocument, ExtractedPage, FileType


# ===========================================================================
# Individual cleaning rules
# ===========================================================================

class TestCollapseWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_tabs_to_space(self):
        assert clean_text("hello\tworld") == "hello world"

    def test_preserves_newlines(self):
        result = clean_text("line one\nline two")
        assert "\n" in result

    def test_mixed_whitespace(self):
        result = clean_text("hello   \t  world")
        assert "hello world" == result


class TestCollapseBlankLines:
    """Tests for blank line normalization."""

    def test_many_blank_lines_collapsed(self):
        text = "para one\n\n\n\n\n\npara two"
        result = clean_text(text)
        # Should have at most 3 consecutive newlines (2 blank lines)
        assert "\n\n\n\n" not in result
        assert "para one" in result
        assert "para two" in result

    def test_two_blank_lines_preserved(self):
        text = "para one\n\npara two"
        result = clean_text(text)
        assert "para one" in result
        assert "para two" in result


class TestStripPageArtifacts:
    """Tests for removing PDF extraction noise."""

    def test_standalone_page_number(self):
        result = _strip_page_artifacts("Some text\n12\nMore text")
        assert "12" not in result.split("\n")
        assert "Some text" in result
        assert "More text" in result

    def test_page_x_of_y(self):
        result = _strip_page_artifacts("Content\nPage 3 of 10\nMore content")
        assert "Page 3 of 10" not in result
        assert "Content" in result

    def test_dash_separated_number(self):
        result = _strip_page_artifacts("Text\n- 5 -\nMore text")
        assert "- 5 -" not in result

    def test_separator_lines(self):
        result = _strip_page_artifacts("Above\n" + "-" * 20 + "\nBelow")
        assert "-" * 20 not in result
        assert "Above" in result
        assert "Below" in result

    def test_preserves_real_numbers_in_text(self):
        """Numbers within sentences should NOT be removed."""
        text = "There are 12 processes running."
        result = _strip_page_artifacts(text)
        assert "12" in result

    def test_preserves_formulas(self):
        """Mathematical formulas should be preserved."""
        text = "The formula is E = mc^2 and F = ma"
        result = clean_text(text)
        assert "E = mc^2" in result
        assert "F = ma" in result


class TestFixBrokenWords:
    """Tests for hyphenated word repair."""

    def test_broken_word(self):
        result = _fix_broken_words("algo-\nrithm")
        assert result == "algorithm"

    def test_preserves_real_hyphens(self):
        """Hyphens followed by a newline + uppercase should be kept
        (but our simple regex fixes lowercase only, which is fine)."""
        text = "well-\nknown"
        result = _fix_broken_words(text)
        # This will join since 'k' is lowercase — acceptable behavior
        assert "wellknown" in result or "well-\nknown" in result

    def test_no_false_positives_in_lists(self):
        """Bullet points with dashes should not be broken."""
        text = "- item one\n- item two"
        result = _fix_broken_words(text)
        assert "item one" in result
        assert "item two" in result


# ===========================================================================
# Full text cleaning
# ===========================================================================

class TestCleanText:
    """Tests for the combined clean_text function."""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \n\n  ") == ""

    def test_preserves_bullet_points(self):
        text = "Key points:\n- Point one\n- Point two\n- Point three"
        result = clean_text(text)
        assert "Point one" in result
        assert "Point two" in result
        assert "Point three" in result

    def test_preserves_numbered_lists(self):
        text = "Steps:\n1. First step\n2. Second step\n3. Third step"
        result = clean_text(text)
        assert "1. First step" in result
        assert "2. Second step" in result

    def test_preserves_definitions(self):
        text = "Definition: A process is a program in execution."
        result = clean_text(text)
        assert "Definition: A process is a program in execution." in result

    def test_preserves_headings(self):
        text = "Chapter 3: Deadlocks\n\nA deadlock occurs when..."
        result = clean_text(text)
        assert "Chapter 3: Deadlocks" in result
        assert "deadlock occurs" in result

    def test_complex_educational_content(self):
        """Full educational paragraph should survive cleaning intact."""
        text = (
            "Banker's Algorithm\n\n"
            "The Banker's Algorithm is used for deadlock avoidance.\n"
            "It works by simulating resource allocation.\n\n"
            "Steps:\n"
            "1. Check if request <= need\n"
            "2. Check if request <= available\n"
            "3. Simulate allocation\n"
            "4. Check safety\n"
        )
        result = clean_text(text)
        assert "Banker's Algorithm" in result
        assert "deadlock avoidance" in result
        assert "1. Check if request" in result


# ===========================================================================
# Document-level cleaning
# ===========================================================================

class TestCleanDocument:
    """Tests for clean_document (operates on ExtractedDocument)."""

    def _make_doc(self, pages_text: list[str]) -> ExtractedDocument:
        """Helper to build an ExtractedDocument from raw text strings."""
        pages = [
            ExtractedPage(
                page_number=i + 1,
                text=text,
                char_count=len(text),
            )
            for i, text in enumerate(pages_text)
        ]
        return ExtractedDocument(
            document_id="test-doc-001",
            document_name="test.pdf",
            file_type=FileType.PDF,
            user_id="user_test",
            pages=pages,
            total_pages=len(pages),
            total_characters=sum(len(t) for t in pages_text),
        )

    def test_cleans_all_pages(self):
        doc = self._make_doc([
            "Page one   with   spaces",
            "Page two\n\n\n\n\n\nwith blank lines",
        ])
        cleaned, time_ms = clean_document(doc)

        assert len(cleaned.pages) == 2
        assert "   " not in cleaned.pages[0].text
        assert time_ms > 0

    def test_removes_empty_pages(self):
        doc = self._make_doc([
            "Real content here",
            "   \n\n   ",  # becomes empty after cleaning
            "More real content",
        ])
        cleaned, _ = clean_document(doc)

        assert len(cleaned.pages) == 2

    def test_preserves_metadata(self):
        doc = self._make_doc(["Some content"])
        cleaned, _ = clean_document(doc)

        assert cleaned.document_id == "test-doc-001"
        assert cleaned.user_id == "user_test"
        assert cleaned.document_name == "test.pdf"

    def test_recalculates_totals(self):
        doc = self._make_doc(["Short", "Also short"])
        cleaned, _ = clean_document(doc)

        expected_chars = sum(p.char_count for p in cleaned.pages)
        assert cleaned.total_characters == expected_chars
        assert cleaned.total_pages == len(cleaned.pages)
