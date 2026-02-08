"""Document indexer service for chunking and embedding documents.

This service handles:
1. Splitting documents into chunks suitable for embedding
2. Generating embeddings for chunks
3. Storing chunks with embeddings in the database
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .embeddings import get_embeddings_service

if TYPE_CHECKING:
    from prisma import Prisma

logger = logging.getLogger(__name__)

# Chunking configuration
# Increased from 512 to 768 for better context per chunk (most embedding models support 1024)
MAX_CHUNK_TOKENS = 768  # Max tokens per chunk
CHUNK_OVERLAP_TOKENS = 80  # Overlap between chunks (increased for better continuity)
MIN_CHUNK_TOKENS = 50  # Minimum tokens for a chunk


@dataclass
class Chunk:
    """A document chunk ready for embedding."""

    content: str
    start_line: int
    end_line: int
    token_count: int
    title: str | None = None


class DocumentIndexer:
    """Service for indexing documents with embeddings for semantic search."""

    def __init__(self, db: Prisma):
        """Initialize the indexer with a database connection."""
        self.db = db
        self.embeddings = get_embeddings_service()

    async def index_document(self, document_id: str) -> int:
        """
        Index a single document by chunking and embedding.

        Args:
            document_id: The document ID to index.

        Returns:
            Number of chunks created.
        """
        # Fetch document
        document = await self.db.document.find_unique(where={"id": document_id})
        if not document:
            logger.warning(f"Document not found: {document_id}")
            return 0

        # Delete existing chunks for this document
        await self.db.execute_raw(
            'DELETE FROM document_chunks WHERE "documentId" = $1',
            document_id,
        )

        # Create chunks
        chunks = self._chunk_document(document.content, document.path)
        if not chunks:
            logger.info(f"No chunks generated for document: {document.path}")
            return 0

        # Generate embeddings for all chunks
        chunk_contents = [c.content for c in chunks]
        embeddings = self.embeddings.embed_texts(chunk_contents)

        # Insert chunks with embeddings using raw SQL (for vector type)
        for chunk, embedding in zip(chunks, embeddings):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await self.db.execute_raw(
                '''
                INSERT INTO document_chunks
                (id, content, embedding, "startLine", "endLine", "tokenCount", title, "createdAt", "documentId")
                VALUES (gen_random_uuid()::text, $1, $2::vector, $3, $4, $5, $6, NOW(), $7)
                ''',
                chunk.content,
                embedding_str,
                chunk.start_line,
                chunk.end_line,
                chunk.token_count,
                chunk.title,
                document_id,
            )

        logger.info(f"Indexed document {document.path}: {len(chunks)} chunks")
        return len(chunks)

    async def index_project(self, project_id: str) -> dict[str, int]:
        """
        Index all documents in a project.

        Args:
            project_id: The project ID to index.

        Returns:
            Dict mapping document paths to chunk counts.
        """
        documents = await self.db.document.find_many(
            where={"projectId": project_id}
        )

        results: dict[str, int] = {}
        for doc in documents:
            chunk_count = await self.index_document(doc.id)
            results[doc.path] = chunk_count

        total_chunks = sum(results.values())
        logger.info(f"Indexed project {project_id}: {len(documents)} docs, {total_chunks} chunks")
        return results

    async def search_similar(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> dict[str, Any]:
        """
        Search for chunks similar to the query using cosine similarity.

        Args:
            project_id: The project to search in.
            query: The search query.
            limit: Maximum number of results.
            min_similarity: Minimum cosine similarity (0-1).

        Returns:
            Dict with 'results' (list of matching chunks) and 'timing' (performance metrics).
        """
        # Time embedding generation (async to avoid blocking event loop)
        embed_start = time.perf_counter()
        query_embedding = await self.embeddings.embed_text_async(query)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)

        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Time vector search query
        search_start = time.perf_counter()
        results = await self.db.query_raw(
            '''
            SELECT
                dc.id,
                dc.content,
                dc."startLine",
                dc."endLine",
                dc."tokenCount",
                dc.title,
                d.path as file_path,
                1 - (dc.embedding <=> $1::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc."documentId" = d.id
            WHERE d."projectId" = $2
              AND 1 - (dc.embedding <=> $1::vector) >= $3
            ORDER BY dc.embedding <=> $1::vector
            LIMIT $4
            ''',
            embedding_str,
            project_id,
            min_similarity,
            limit,
        )
        search_ms = int((time.perf_counter() - search_start) * 1000)

        # Log timing for monitoring
        logger.info(
            f"vector_search: project={project_id} results={len(results)} "
            f"embed_ms={embed_ms} search_ms={search_ms} total_ms={embed_ms + search_ms}"
        )

        return {
            "results": [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "start_line": row["startLine"],
                    "end_line": row["endLine"],
                    "token_count": row["tokenCount"],
                    "title": row["title"],
                    "file_path": row["file_path"],
                    "similarity": float(row["similarity"]),
                }
                for row in results
            ],
            "timing": {
                "embed_ms": embed_ms,
                "search_ms": search_ms,
                "total_ms": embed_ms + search_ms,
            },
        }

    def _chunk_document(self, content: str, file_path: str) -> list[Chunk]:
        """
        Split document content into chunks suitable for embedding.

        Uses a markdown-aware chunking strategy:
        1. First split by headers to preserve section context
        2. Then split large sections by paragraphs
        3. Finally split very long paragraphs by sentences
        """
        lines = content.split("\n")
        chunks: list[Chunk] = []

        # First pass: split by headers
        sections = self._split_by_headers(lines)

        for section_start, section_end, section_title, section_lines in sections:
            section_content = "\n".join(section_lines)
            section_tokens = self._estimate_tokens(section_content)

            if section_tokens <= MAX_CHUNK_TOKENS:
                # Section fits in one chunk
                if section_tokens >= MIN_CHUNK_TOKENS:
                    chunks.append(Chunk(
                        content=section_content,
                        start_line=section_start,
                        end_line=section_end,
                        token_count=section_tokens,
                        title=section_title,
                    ))
            else:
                # Section too large - split by paragraphs
                paragraph_chunks = self._split_section_by_paragraphs(
                    section_lines, section_start, section_title
                )
                chunks.extend(paragraph_chunks)

        return chunks

    def _split_by_headers(
        self, lines: list[str]
    ) -> list[tuple[int, int, str | None, list[str]]]:
        """
        Split document into sections by markdown headers.

        Returns list of (start_line, end_line, title, lines) tuples.
        """
        sections: list[tuple[int, int, str | None, list[str]]] = []
        current_start = 1
        current_title: str | None = None
        current_lines: list[str] = []
        in_code_block = False

        for i, line in enumerate(lines, start=1):
            # Track fenced code blocks to avoid parsing comments as headers
            if line.startswith("```") or line.startswith("~~~"):
                in_code_block = not in_code_block

            # Only match headers outside code blocks
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code_block else None

            if header_match:
                # Save previous section if non-empty
                if current_lines:
                    sections.append((
                        current_start,
                        i - 1,
                        current_title,
                        current_lines,
                    ))

                # Start new section
                current_start = i
                current_title = header_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections.append((
                current_start,
                len(lines),
                current_title,
                current_lines,
            ))

        return sections

    def _split_section_by_paragraphs(
        self, lines: list[str], start_offset: int, section_title: str | None
    ) -> list[Chunk]:
        """Split a large section by paragraphs, then by sentences if needed."""
        chunks: list[Chunk] = []
        current_chunk_lines: list[str] = []
        current_start = start_offset

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            current_content = "\n".join(current_chunk_lines)
            current_tokens = self._estimate_tokens(current_content)

            # Check if we should end the chunk
            is_paragraph_break = line.strip() == "" and i > 0
            is_approaching_limit = current_tokens >= MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS

            # Force split if we've significantly exceeded the limit (even mid-paragraph)
            force_split = current_tokens >= MAX_CHUNK_TOKENS + 100

            if (is_paragraph_break and is_approaching_limit) or force_split:
                # If force splitting, try to end at sentence boundary
                if force_split and not is_paragraph_break:
                    current_content = self._truncate_at_sentence(current_content, MAX_CHUNK_TOKENS)
                    current_tokens = self._estimate_tokens(current_content)

                # Create chunk
                if current_tokens >= MIN_CHUNK_TOKENS:
                    chunks.append(Chunk(
                        content=current_content.strip(),
                        start_line=current_start,
                        end_line=start_offset + i,
                        token_count=current_tokens,
                        title=section_title,
                    ))

                # Start new chunk with overlap
                overlap_lines = current_chunk_lines[-3:] if len(current_chunk_lines) > 3 else []
                current_chunk_lines = overlap_lines
                current_start = start_offset + i - len(overlap_lines) + 1

        # Handle remaining content
        if current_chunk_lines:
            current_content = "\n".join(current_chunk_lines)
            current_tokens = self._estimate_tokens(current_content)

            if current_tokens >= MIN_CHUNK_TOKENS:
                chunks.append(Chunk(
                    content=current_content.strip(),
                    start_line=current_start,
                    end_line=start_offset + len(lines) - 1,
                    token_count=current_tokens,
                    title=section_title,
                ))
            elif chunks and current_tokens > 0:
                # Merge with previous chunk if too small
                last_chunk = chunks[-1]
                merged_content = last_chunk.content + "\n\n" + current_content.strip()
                chunks[-1] = Chunk(
                    content=merged_content,
                    start_line=last_chunk.start_line,
                    end_line=start_offset + len(lines) - 1,
                    token_count=self._estimate_tokens(merged_content),
                    title=section_title,
                )

        return chunks

    def _truncate_at_sentence(self, content: str, max_tokens: int) -> str:
        """Truncate content at sentence boundary within token budget."""
        max_chars = max_tokens * 4  # Rough estimate: 4 chars per token

        if len(content) <= max_chars:
            return content

        # Find last sentence ending before max_chars
        truncated = content[:max_chars]
        sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
        best_end = -1

        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > best_end and pos > len(truncated) * 0.5:
                best_end = pos + len(ending)

        if best_end > 0:
            return truncated[:best_end].rstrip()

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > len(truncated) * 0.7:
            return truncated[:last_space].rstrip()

        return truncated.rstrip()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        # This is a fast approximation. For exact counts, use tiktoken.
        return len(text) // 4


async def get_indexer(db: Prisma) -> DocumentIndexer:
    """Create a document indexer instance."""
    return DocumentIndexer(db)
