"""Tests for rlm_context_query tool - Token budgeting and context optimization."""

import pytest

from src.models import (
    ContextQueryParams,
    ContextQueryResult,
    ContextSection,
    SearchMode,
    ToolName,
)
from src.rlm_engine import RLMEngine, Section, count_tokens, get_encoder


class TestTokenCounting:
    """Tests for tiktoken integration."""

    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 10  # Should be ~4 tokens

    def test_count_tokens_empty(self):
        """Test token counting for empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        """Test token counting for longer text."""
        text = "This is a longer piece of text that should have more tokens. " * 10
        tokens = count_tokens(text)
        assert tokens > 50
        assert tokens < 200

    def test_encoder_singleton(self):
        """Test that encoder is lazily initialized and reused."""
        enc1 = get_encoder()
        enc2 = get_encoder()
        assert enc1 is enc2  # Same instance


class TestContextQueryParams:
    """Tests for ContextQueryParams validation."""

    def test_default_values(self):
        """Test default parameter values."""
        params = ContextQueryParams(query="test query")
        assert params.query == "test query"
        assert params.max_tokens == 4000
        assert params.search_mode == SearchMode.KEYWORD
        assert params.include_metadata is True

    def test_custom_values(self):
        """Test custom parameter values."""
        params = ContextQueryParams(
            query="How does auth work?",
            max_tokens=8000,
            search_mode=SearchMode.SEMANTIC,
            include_metadata=False,
        )
        assert params.query == "How does auth work?"
        assert params.max_tokens == 8000
        assert params.search_mode == SearchMode.SEMANTIC
        assert params.include_metadata is False

    def test_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        # Valid minimum
        params = ContextQueryParams(query="test", max_tokens=100)
        assert params.max_tokens == 100

        # Valid maximum
        params = ContextQueryParams(query="test", max_tokens=100000)
        assert params.max_tokens == 100000

        # Below minimum should fail
        with pytest.raises(ValueError):
            ContextQueryParams(query="test", max_tokens=50)

        # Above maximum should fail
        with pytest.raises(ValueError):
            ContextQueryParams(query="test", max_tokens=200000)


class TestContextSection:
    """Tests for ContextSection model."""

    def test_section_creation(self):
        """Test creating a context section."""
        section = ContextSection(
            title="Authentication",
            content="This section covers authentication...",
            file="docs/auth.md",
            lines=(10, 50),
            relevance_score=0.85,
            token_count=150,
            truncated=False,
        )
        assert section.title == "Authentication"
        assert section.file == "docs/auth.md"
        assert section.lines == (10, 50)
        assert section.relevance_score == 0.85
        assert section.token_count == 150
        assert section.truncated is False

    def test_truncated_section(self):
        """Test creating a truncated section."""
        section = ContextSection(
            title="Long Section",
            content="Content that was truncated...",
            file="docs/long.md",
            lines=(1, 1000),
            relevance_score=0.92,
            token_count=500,
            truncated=True,
        )
        assert section.truncated is True

    def test_relevance_score_bounds(self):
        """Test relevance score validation."""
        # Valid scores
        ContextSection(
            title="Test",
            content="content",
            file="test.md",
            lines=(1, 10),
            relevance_score=0.0,
            token_count=10,
        )
        ContextSection(
            title="Test",
            content="content",
            file="test.md",
            lines=(1, 10),
            relevance_score=1.0,
            token_count=10,
        )

        # Invalid scores
        with pytest.raises(ValueError):
            ContextSection(
                title="Test",
                content="content",
                file="test.md",
                lines=(1, 10),
                relevance_score=-0.1,
                token_count=10,
            )
        with pytest.raises(ValueError):
            ContextSection(
                title="Test",
                content="content",
                file="test.md",
                lines=(1, 10),
                relevance_score=1.5,
                token_count=10,
            )


class TestContextQueryResult:
    """Tests for ContextQueryResult model."""

    def test_empty_result(self):
        """Test creating an empty result."""
        result = ContextQueryResult(
            sections=[],
            total_tokens=0,
            max_tokens=4000,
            query="no matches",
            search_mode=SearchMode.KEYWORD,
        )
        assert len(result.sections) == 0
        assert result.total_tokens == 0
        assert result.session_context_included is False
        assert len(result.suggestions) == 0

    def test_result_with_sections(self):
        """Test creating a result with sections."""
        section = ContextSection(
            title="Test",
            content="content",
            file="test.md",
            lines=(1, 10),
            relevance_score=0.8,
            token_count=50,
        )
        result = ContextQueryResult(
            sections=[section],
            total_tokens=50,
            max_tokens=4000,
            query="test query",
            search_mode=SearchMode.KEYWORD,
            session_context_included=True,
            suggestions=["Other Section (score: 5.0)"],
        )
        assert len(result.sections) == 1
        assert result.total_tokens == 50
        assert result.session_context_included is True
        assert len(result.suggestions) == 1


class TestKeywordScoring:
    """Tests for keyword-based relevance scoring."""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing."""
        return RLMEngine("test-project")

    def test_title_weight(self, engine):
        """Test that title matches are weighted higher."""
        section_title_match = Section(
            id="[AUTH]",
            title="Authentication Flow",
            content="Some content about login",
            start_line=1,
            end_line=10,
            level=1,
        )
        section_content_match = Section(
            id="[OTHER]",
            title="Other Section",
            content="Content about authentication and login flow",
            start_line=11,
            end_line=20,
            level=2,
        )

        keywords = ["authentication", "flow"]
        score_title = engine._calculate_keyword_score(section_title_match, keywords)
        score_content = engine._calculate_keyword_score(section_content_match, keywords)

        # Title match should score higher
        assert score_title > score_content

    def test_level_bonus(self, engine):
        """Test that higher-level sections get bonus points."""
        section_h1 = Section(
            id="[H1]",
            title="Main Topic",
            content="Content with keyword",
            start_line=1,
            end_line=10,
            level=1,
        )
        section_h3 = Section(
            id="[H3]",
            title="Main Topic",
            content="Content with keyword",
            start_line=1,
            end_line=10,
            level=3,
        )

        keywords = ["keyword"]
        score_h1 = engine._calculate_keyword_score(section_h1, keywords)
        score_h3 = engine._calculate_keyword_score(section_h3, keywords)

        # H1 should score higher than H3
        assert score_h1 > score_h3

    def test_no_match_zero_score(self, engine):
        """Test that non-matching sections get zero score."""
        section = Section(
            id="[TEST]",
            title="Unrelated Topic",
            content="Nothing relevant here",
            start_line=1,
            end_line=10,
            level=2,
        )

        keywords = ["authentication", "login"]
        score = engine._calculate_keyword_score(section, keywords)

        assert score == 0.0


class TestSmartTruncation:
    """Tests for smart content truncation."""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing."""
        return RLMEngine("test-project")

    def test_no_truncation_needed(self, engine):
        """Test that short content is not truncated."""
        content = "Short content."
        result = engine._smart_truncate(content, 100)
        assert result == content
        assert "..." not in result

    def test_truncate_at_sentence(self, engine):
        """Test that truncation happens at sentence boundary."""
        content = "First sentence. Second sentence. Third sentence that is longer."
        # Set a token limit that will require truncation
        result = engine._smart_truncate(content, 10)

        # Should end with ... and at a sentence boundary
        assert result.endswith("...")
        # Should preserve at least some content
        assert len(result) > 10

    def test_truncate_preserves_meaning(self, engine):
        """Test that truncation preserves as much content as possible."""
        content = "Important information here. More details follow. Even more content."
        result = engine._smart_truncate(content, 15)

        # Should contain the beginning of the content
        assert result.startswith("Important")
        assert result.endswith("...")


class TestSearchModeEnum:
    """Tests for SearchMode enum."""

    def test_valid_modes(self):
        """Test valid search modes."""
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.HYBRID.value == "hybrid"

    def test_enum_conversion(self):
        """Test converting strings to SearchMode."""
        assert SearchMode("keyword") == SearchMode.KEYWORD
        assert SearchMode("semantic") == SearchMode.SEMANTIC
        assert SearchMode("hybrid") == SearchMode.HYBRID

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError):
            SearchMode("invalid")


class TestToolNameEnum:
    """Tests for ToolName enum including new context_query."""

    def test_context_query_in_enum(self):
        """Test that rlm_context_query is in the ToolName enum."""
        assert ToolName.RLM_CONTEXT_QUERY.value == "rlm_context_query"

    def test_all_tools_present(self):
        """Test that all expected tools are present."""
        expected_tools = [
            "rlm_ask",
            "rlm_search",
            "rlm_inject",
            "rlm_context",
            "rlm_clear_context",
            "rlm_stats",
            "rlm_sections",
            "rlm_read",
            "rlm_context_query",
        ]
        tool_values = [t.value for t in ToolName]
        for tool in expected_tools:
            assert tool in tool_values, f"Missing tool: {tool}"
