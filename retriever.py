0"""
retriever.py
────────────
Semantic retrieval layer for ChefMind AI.

Given a user cooking query, this module:
1. Opens the persisted Chroma collection (built by ingest.py).
2. Embeds the query with the same IBM watsonx embedding model.
3. Returns the top-K most relevant recipe chunks as a single context string.
"""

from __future__ import annotations

import os
import sys

from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from rich.console import Console

import config

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton vector store (loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """Load (or return cached) the persistent Chroma vector store."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        console.print(
            "[red]✖  Chroma store not found.[/red]\n"
            "    Run [bold]python ingest.py[/bold] first to build the vector index."
        )
        sys.exit(1)

    embeddings = WatsonxEmbeddings(
        model_id=config.EMBEDDING_MODEL_ID,
        url=config.WATSONX_URL,
        apikey=config.WATSONX_API_KEY,
        project_id=config.WATSONX_PROJECT_ID,
    )

    _vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    return _vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = config.TOP_K_RESULTS) -> str:
    """
    Retrieve the most relevant recipe chunks for *query*.

    Returns
    ───────
    A single string containing the retrieved chunks separated by
    horizontal rules — ready to be injected into the ChefMind prompt.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)

    if not results:
        return "No relevant recipe context found in the knowledge base."

    parts: list[str] = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown source")
        filename = os.path.basename(source)
        parts.append(f"[Chunk {i} — {filename}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
