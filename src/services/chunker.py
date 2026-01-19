"""Smart document chunking service.

Chunks documents respecting markdown structure, code blocks, and semantic boundaries.
This is Phase 4.4 implementation.
"""

import re
from dataclasses import dataclass, field

import tiktoken

# Lazy-load encoder
_encoding: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(get_encoder().encode(text))


@dataclass
class Chunk:
    """A document chunk."""

    content: str
    header: str = ""
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    is_code_block: bool = False

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ChunkingResult:
    """Result of chunking a document."""

    chunks: list[Chunk] = field(default_factory=list)
    total_tokens: int = 0
    total_chunks: int = 0


class DocumentChunker:
    """Smart document chunker that respects markdown structure."""

    # Stop words to filter out when extracting terms
    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and", "but",
        "if", "or", "because", "until", "while", "although", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "i", "me",
        "my", "myself", "we", "our", "ours", "you", "your", "he", "him",
        "she", "her", "it", "its", "they", "them", "their", "explain",
        "describe", "help", "tell", "show", "make", "get", "find", "work",
    })

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 50,
    ):
        """Initialize the chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks (for context)
            min_chunk_tokens: Minimum tokens for a valid chunk
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_document(self, content: str, file_path: str = "") -> ChunkingResult:
        """
        Chunk a document respecting markdown structure.

        Rules:
        1. Never split in the middle of a code block
        2. Prefer splitting at header boundaries
        3. Fall back to paragraph boundaries
        4. Last resort: sentence boundaries
        """
        chunks: list[Chunk] = []
        lines = content.split("\n")

        # First pass: identify code blocks and headers
        code_block_ranges = self._find_code_blocks(lines)
        header_lines = self._find_headers(lines)

        # Build sections based on headers
        sections = self._build_sections(lines, header_lines, code_block_ranges)

        # Process each section
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)

        # Calculate totals
        total_tokens = sum(c.token_count for c in chunks)

        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            total_chunks=len(chunks),
        )

    def _find_code_blocks(self, lines: list[str]) -> list[tuple[int, int]]:
        """Find code block ranges (start, end line indices)."""
        ranges = []
        in_code_block = False
        start = 0

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if in_code_block:
                    ranges.append((start, i))
                    in_code_block = False
                else:
                    start = i
                    in_code_block = True

        return ranges

    def _find_headers(self, lines: list[str]) -> list[tuple[int, int, str]]:
        """Find header lines with (line_idx, level, title)."""
        headers = []
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append((i, level, title))

        return headers

    def _is_in_code_block(
        self, line_idx: int, code_blocks: list[tuple[int, int]]
    ) -> bool:
        """Check if a line is inside a code block."""
        for start, end in code_blocks:
            if start <= line_idx <= end:
                return True
        return False

    def _build_sections(
        self,
        lines: list[str],
        headers: list[tuple[int, int, str]],
        code_blocks: list[tuple[int, int]],
    ) -> list[dict]:
        """Build sections from headers."""
        if not headers:
            # No headers - treat entire document as one section
            return [{
                "header": "",
                "content": "\n".join(lines),
                "start_line": 0,
                "end_line": len(lines) - 1,
            }]

        sections = []
        for i, (line_idx, level, title) in enumerate(headers):
            # Find end of section (next header of same or higher level, or EOF)
            end_idx = len(lines) - 1
            for j in range(i + 1, len(headers)):
                next_idx, next_level, _ = headers[j]
                if next_level <= level:
                    end_idx = next_idx - 1
                    break

            section_lines = lines[line_idx : end_idx + 1]
            sections.append({
                "header": title,
                "header_line": lines[line_idx],
                "content": "\n".join(section_lines),
                "start_line": line_idx,
                "end_line": end_idx,
                "level": level,
            })

        return sections

    def _chunk_section(self, section: dict) -> list[Chunk]:
        """Chunk a single section."""
        content = section["content"]
        header = section.get("header", "")
        start_line = section.get("start_line", 0)

        # If section fits in one chunk, return as-is
        if count_tokens(content) <= self.max_chunk_tokens:
            return [Chunk(
                content=content,
                header=header,
                start_line=start_line,
                end_line=section.get("end_line", start_line),
            )]

        # Need to split - try paragraphs first
        return self._split_by_paragraphs(content, header, start_line)

    def _split_by_paragraphs(
        self, content: str, header: str, start_line: int
    ) -> list[Chunk]:
        """Split content by paragraphs."""
        chunks = []

        # Split by double newline (paragraphs)
        paragraphs = re.split(r"\n\n+", content)
        current_chunk_parts: list[str] = []
        current_tokens = 0
        chunk_start_line = start_line

        # Always include header in first chunk
        header_line = ""
        if header:
            for para in paragraphs:
                if para.strip().startswith("#") and header in para:
                    header_line = para
                    break

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens <= self.max_chunk_tokens:
                current_chunk_parts.append(para)
                current_tokens += para_tokens
            else:
                # Save current chunk if it has content
                if current_chunk_parts:
                    chunk_content = "\n\n".join(current_chunk_parts)
                    # Prepend header if this is first chunk
                    if not chunks and header_line and header_line not in chunk_content:
                        chunk_content = header_line + "\n\n" + chunk_content

                    chunks.append(Chunk(
                        content=chunk_content,
                        header=header,
                        start_line=chunk_start_line,
                    ))
                    chunk_start_line += chunk_content.count("\n") + 1

                # Start new chunk
                if para_tokens <= self.max_chunk_tokens:
                    # Add header context if not the first chunk
                    if header_line and chunks:
                        current_chunk_parts = [f"{header_line}\n\n(continued...)", para]
                        current_tokens = count_tokens("\n\n".join(current_chunk_parts))
                    else:
                        current_chunk_parts = [para]
                        current_tokens = para_tokens
                else:
                    # Paragraph too big - split by sentences
                    sentence_chunks = self._split_by_sentences(para, header, chunk_start_line)
                    chunks.extend(sentence_chunks)
                    current_chunk_parts = []
                    current_tokens = 0

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk_content = "\n\n".join(current_chunk_parts)
            if not chunks and header_line and header_line not in chunk_content:
                chunk_content = header_line + "\n\n" + chunk_content
            chunks.append(Chunk(
                content=chunk_content,
                header=header,
                start_line=chunk_start_line,
            ))

        return chunks

    def _split_by_sentences(
        self, content: str, header: str, start_line: int
    ) -> list[Chunk]:
        """Split content by sentences as last resort."""
        chunks = []

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", content)
        current_chunk_parts: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if current_tokens + sentence_tokens <= self.max_chunk_tokens:
                current_chunk_parts.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk_parts:
                    chunks.append(Chunk(
                        content=" ".join(current_chunk_parts),
                        header=header,
                        start_line=start_line,
                    ))

                if sentence_tokens <= self.max_chunk_tokens:
                    current_chunk_parts = [sentence]
                    current_tokens = sentence_tokens
                else:
                    # Even a single sentence is too long - force split by tokens
                    chunks.extend(self._force_split(sentence, header, start_line))
                    current_chunk_parts = []
                    current_tokens = 0

        if current_chunk_parts:
            chunks.append(Chunk(
                content=" ".join(current_chunk_parts),
                header=header,
                start_line=start_line,
            ))

        return chunks

    def _force_split(self, content: str, header: str, start_line: int) -> list[Chunk]:
        """Force split content by token count (last resort)."""
        chunks = []
        encoder = get_encoder()
        tokens = encoder.encode(content)

        for i in range(0, len(tokens), self.max_chunk_tokens - self.overlap_tokens):
            chunk_tokens = tokens[i : i + self.max_chunk_tokens]
            chunk_content = encoder.decode(chunk_tokens)

            if len(chunk_content.strip()) > 0:
                chunks.append(Chunk(
                    content=chunk_content + "...",
                    header=header,
                    start_line=start_line,
                    token_count=len(chunk_tokens),
                ))

        return chunks

    def extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text (no LLM required).

        Args:
            text: The text to extract terms from

        Returns:
            List of key terms sorted by relevance
        """
        # Tokenize and clean
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text.lower())

        # Filter stop words and short words
        terms = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Count frequency
        term_counts: dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        # Find bigrams
        bigrams: list[str] = []
        for i in range(len(terms) - 1):
            bigram = f"{terms[i]} {terms[i + 1]}"
            bigrams.append(bigram)

        bigram_counts: dict[str, int] = {}
        for bigram in bigrams:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Combine and sort by frequency
        all_terms = list(term_counts.items()) + list(bigram_counts.items())
        all_terms.sort(key=lambda x: x[1], reverse=True)

        # Return top terms (deduplicated)
        seen = set()
        result = []
        for term, _ in all_terms:
            if term not in seen:
                result.append(term)
                seen.add(term)
            if len(result) >= 20:
                break

        return result


# Singleton instance
_chunker: DocumentChunker | None = None


def get_chunker() -> DocumentChunker:
    """Get the singleton chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = DocumentChunker()
    return _chunker
