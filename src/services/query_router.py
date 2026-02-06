"""Smart Query Router - Automatically selects optimal execution mode.

This module provides intelligent routing for Snipara queries, determining
whether to use direct context (Mode B) or RLM-Runtime (Mode C) based on
query characteristics.

Routing Criteria:
- Simple factual queries → Direct context (fast, token-efficient)
- Multi-step reasoning → RLM-Runtime (better quality)
- Code generation → RLM-Runtime (iterative refinement)
- Complex queries needing decomposition → RLM-Runtime

Usage:
    from services.query_router import QueryRouter, RouteDecision

    router = QueryRouter()
    decision = router.route(query, context_tokens=4000)

    if decision.mode == "direct":
        # Use rlm_context_query directly
        result = await context_query(query, max_tokens=decision.budget)
    else:
        # Use RLM-Runtime
        result = await rlm.completion(query, options=decision.rlm_options)
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryMode(Enum):
    """Execution modes for queries."""
    DIRECT = "direct"       # Direct context query (Mode B)
    RLM_RUNTIME = "rlm"     # RLM-Runtime with tools (Mode C)


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"       # Single-topic factual queries
    MODERATE = "moderate"   # Multi-aspect queries
    COMPLEX = "complex"     # Multi-step reasoning, code generation


@dataclass
class RouteDecision:
    """Routing decision with execution parameters."""
    mode: QueryMode
    complexity: QueryComplexity
    confidence: float  # 0-1, how confident we are in this routing
    reason: str        # Human-readable explanation

    # Mode-specific parameters
    token_budget: int = 6000     # Context token budget
    search_mode: str = "hybrid"  # keyword, semantic, hybrid
    rlm_max_depth: int = 3       # Max RLM recursion depth
    rlm_token_budget: int = 30000  # RLM total token budget

    @property
    def is_direct(self) -> bool:
        return self.mode == QueryMode.DIRECT

    @property
    def is_rlm(self) -> bool:
        return self.mode == QueryMode.RLM_RUNTIME


@dataclass
class QueryRouter:
    """Smart query router that selects optimal execution mode.

    The router analyzes query characteristics to determine:
    1. Query complexity (simple/moderate/complex)
    2. Best execution mode (direct context vs RLM-Runtime)
    3. Optimal parameters (token budgets, search mode, etc.)

    Routing heuristics:
    - Code-related queries → RLM (needs iterative refinement)
    - "How to" / "Explain" queries → RLM (needs reasoning)
    - Simple lookups ("What is X?") → Direct (fast, efficient)
    - Multi-part queries → RLM (needs decomposition)
    """

    # Patterns that indicate complex queries needing RLM
    _COMPLEX_PATTERNS: list[re.Pattern] = field(default_factory=list)

    # Patterns that indicate simple queries (prefer direct)
    _SIMPLE_PATTERNS: list[re.Pattern] = field(default_factory=list)

    # Code-related keywords
    _CODE_KEYWORDS: set[str] = field(default_factory=set)

    def __post_init__(self):
        # Complex query patterns → RLM-Runtime
        self._COMPLEX_PATTERNS = [
            re.compile(r"\b(how to|how do|how can|explain|describe|walk me through)\b", re.I),
            re.compile(r"\b(implement|create|build|write|generate|refactor)\b.*\b(code|function|class|component|api|endpoint)\b", re.I),
            re.compile(r"\b(step by step|steps to|process for)\b", re.I),
            re.compile(r"\b(compare|difference between|versus|vs\.?)\b", re.I),
            re.compile(r"\b(debug|fix|solve|troubleshoot)\b", re.I),
            re.compile(r"\b(and|also|additionally|furthermore)\b.*\?", re.I),  # Multi-part questions
            re.compile(r"\?.*\?", re.I),  # Multiple questions
        ]

        # Simple query patterns → Direct context
        self._SIMPLE_PATTERNS = [
            re.compile(r"^what (is|are) (the )?(.*?)\??$", re.I),
            re.compile(r"^where (is|are|can|do)\b", re.I),
            re.compile(r"^which\b", re.I),
            re.compile(r"^(list|show|get) (the )?(.*?)\??$", re.I),
            re.compile(r"\b(pricing|plans?|cost|price)\b", re.I),
            re.compile(r"\b(version|release)\b", re.I),
        ]

        # Code-related keywords
        self._CODE_KEYWORDS = {
            "code", "function", "class", "method", "api", "endpoint",
            "implement", "create", "build", "write", "generate", "refactor",
            "typescript", "javascript", "python", "react", "nextjs",
            "test", "unit test", "integration test", "lint", "typecheck",
        }

    def route(
        self,
        query: str,
        context_tokens: int = 0,
        force_mode: QueryMode | None = None,
    ) -> RouteDecision:
        """Determine optimal execution mode for a query.

        Args:
            query: The user's query
            context_tokens: Current context size (if available)
            force_mode: Override routing decision (for testing)

        Returns:
            RouteDecision with mode, parameters, and reasoning
        """
        if force_mode:
            return RouteDecision(
                mode=force_mode,
                complexity=QueryComplexity.MODERATE,
                confidence=1.0,
                reason=f"Forced to {force_mode.value} mode",
            )

        # Analyze query
        complexity = self._assess_complexity(query)
        code_related = self._is_code_related(query)
        multi_part = self._is_multi_part(query)

        # Decision matrix
        if complexity == QueryComplexity.COMPLEX or code_related or multi_part:
            return self._route_to_rlm(query, complexity, code_related, multi_part)
        elif complexity == QueryComplexity.SIMPLE:
            return self._route_to_direct(query, complexity)
        else:
            # Moderate complexity - use context size as tiebreaker
            if context_tokens > 10000:
                # Large context already available, use direct
                return self._route_to_direct(query, complexity)
            else:
                # Let RLM fetch optimal context
                return self._route_to_rlm(query, complexity, code_related, multi_part)

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on patterns."""
        # Check for complex patterns
        complex_matches = sum(1 for p in self._COMPLEX_PATTERNS if p.search(query))
        if complex_matches >= 2:
            return QueryComplexity.COMPLEX

        # Check for simple patterns
        simple_matches = sum(1 for p in self._SIMPLE_PATTERNS if p.search(query))
        if simple_matches >= 1 and complex_matches == 0:
            return QueryComplexity.SIMPLE

        # Word count heuristic
        word_count = len(query.split())
        if word_count <= 8:
            return QueryComplexity.SIMPLE
        elif word_count >= 25:
            return QueryComplexity.COMPLEX

        return QueryComplexity.MODERATE

    def _is_code_related(self, query: str) -> bool:
        """Check if query is code-related."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self._CODE_KEYWORDS)

    def _is_multi_part(self, query: str) -> bool:
        """Check if query has multiple parts/questions."""
        # Multiple question marks
        if query.count("?") >= 2:
            return True
        # Conjunctions suggesting multiple requests
        if re.search(r"\b(and|also|additionally)\b.*\b(and|also|additionally)\b", query, re.I):
            return True
        # Numbered or bulleted items
        if re.search(r"(^|\n)\s*[1-9]\.", query):
            return True
        return False

    def _route_to_direct(self, query: str, complexity: QueryComplexity) -> RouteDecision:
        """Create decision for direct context mode."""
        # Select search mode based on query
        if re.search(r"\b(exact|specific|called|named)\b", query, re.I):
            search_mode = "keyword"
        elif re.search(r"\b(like|similar|related|about)\b", query, re.I):
            search_mode = "semantic"
        else:
            search_mode = "hybrid"

        return RouteDecision(
            mode=QueryMode.DIRECT,
            complexity=complexity,
            confidence=0.8 if complexity == QueryComplexity.SIMPLE else 0.6,
            reason=f"Simple {complexity.value} query - direct context is efficient",
            token_budget=6000,
            search_mode=search_mode,
        )

    def _route_to_rlm(
        self,
        query: str,
        complexity: QueryComplexity,
        code_related: bool,
        multi_part: bool,
    ) -> RouteDecision:
        """Create decision for RLM-Runtime mode."""
        reasons = []
        if complexity == QueryComplexity.COMPLEX:
            reasons.append("complex reasoning required")
        if code_related:
            reasons.append("code-related task")
        if multi_part:
            reasons.append("multi-part question")

        # Adjust parameters based on complexity
        if complexity == QueryComplexity.COMPLEX:
            rlm_budget = 50000
            max_depth = 5
        else:
            rlm_budget = 30000
            max_depth = 3

        return RouteDecision(
            mode=QueryMode.RLM_RUNTIME,
            complexity=complexity,
            confidence=0.85 if code_related else 0.7,
            reason=f"RLM recommended: {', '.join(reasons)}",
            token_budget=8000,  # Pre-fetch more context for RLM
            search_mode="hybrid",
            rlm_max_depth=max_depth,
            rlm_token_budget=rlm_budget,
        )


# Singleton instance for convenience
_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    """Get or create the singleton router instance."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def route_query(query: str, context_tokens: int = 0) -> RouteDecision:
    """Convenience function to route a query.

    Args:
        query: The user's query
        context_tokens: Current context size (optional)

    Returns:
        RouteDecision with optimal mode and parameters
    """
    return get_router().route(query, context_tokens)


# Query complexity scoring for external use
def assess_query_complexity(query: str) -> dict:
    """Assess query complexity and return detailed analysis.

    Useful for debugging routing decisions or logging.

    Returns:
        {
            "complexity": "simple" | "moderate" | "complex",
            "code_related": bool,
            "multi_part": bool,
            "word_count": int,
            "recommended_mode": "direct" | "rlm",
        }
    """
    router = get_router()
    decision = router.route(query)

    return {
        "complexity": decision.complexity.value,
        "code_related": router._is_code_related(query),
        "multi_part": router._is_multi_part(query),
        "word_count": len(query.split()),
        "recommended_mode": decision.mode.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
    }
