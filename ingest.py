"""
ingest.py
─────────
Document ingestion pipeline for ChefMind AI.

Responsibilities
────────────────
1. Load recipe documents from RECIPES_DIR (plain-text .txt files or .md files).
2. Split them into overlapping chunks with LangChain's RecursiveCharacterTextSplitter.
3. Embed each chunk using HuggingFace (USE_LOCAL=true) or IBM watsonx (USE_LOCAL=false).
4. Persist all vectors in a local Chroma collection.

Usage
─────
    python ingest.py                        # ingest everything in data/recipes/
    python ingest.py --recipes-dir /path    # custom directory
    python ingest.py --reset                # wipe and re-build the collection
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from rich.console import Console
from rich.progress import track

import config

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_documents(recipes_dir: str) -> list:
    """Load all .txt and .md files from *recipes_dir* as LangChain Documents."""
    patterns = [
        os.path.join(recipes_dir, "**", "*.txt"),
        os.path.join(recipes_dir, "**", "*.md"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))

    if not paths:
        console.print(
            f"[yellow]⚠  No .txt or .md recipe files found in '{recipes_dir}'.[/yellow]\n"
            "    Add recipe files and re-run ingest.py."
        )
        sys.exit(0)

    docs = []
    for path in track(paths, description="[cyan]Loading files…[/cyan]"):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    console.print(f"[green]✔  Loaded {len(docs)} document(s) from {len(paths)} file(s).[/green]")
    return docs


def _split_documents(docs: list) -> list:
    """Chunk documents into overlapping text segments."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[green]✔  Split into {len(chunks)} chunk(s) "
                  f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).[/green]")
    return chunks


def _build_embeddings():
    """Return HuggingFace embeddings (local) or IBM watsonx embeddings."""
    if config.USE_LOCAL:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        console.print(
            f"[dim]Embedding model: {config.LOCAL_EMBEDDING_MODEL} (local HuggingFace)[/dim]"
        )
        return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)
    else:
        from langchain_ibm import WatsonxEmbeddings
        console.print(
            f"[dim]Embedding model: {config.WATSONX_EMBEDDING_MODEL_ID} (IBM watsonx)[/dim]"
        )
        return WatsonxEmbeddings(
            model_id=config.WATSONX_EMBEDDING_MODEL_ID,
            url=config.WATSONX_URL,
            apikey=config.WATSONX_API_KEY,
            project_id=config.WATSONX_PROJECT_ID,
        )


def _persist_to_chroma(chunks: list, embeddings, reset: bool) -> Chroma:
    """
    Store embedded chunks in a persistent Chroma collection.

    If *reset* is True, delete the existing collection first.
    """
    if reset and os.path.exists(config.CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        console.print("[yellow]↺  Existing Chroma collection wiped.[/yellow]")

    console.print("[cyan]Embedding and persisting chunks — this may take a moment…[/cyan]")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    console.print(
        f"[green]✔  {len(chunks)} chunks stored in Chroma "
        f"at '{config.CHROMA_PERSIST_DIR}'.[/green]"
    )
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(recipes_dir: str = config.RECIPES_DIR, reset: bool = False) -> None:
    """Full ingestion pipeline: load → split → embed → store."""
    console.rule("[bold blue]ChefMind AI — Document Ingestion[/bold blue]")

    docs = _load_documents(recipes_dir)
    chunks = _split_documents(docs)
    embeddings = _build_embeddings()
    _persist_to_chroma(chunks, embeddings, reset=reset)

    console.rule("[bold green]Ingestion complete[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChefMind AI — ingest recipe documents")
    parser.add_argument(
        "--recipes-dir",
        default=config.RECIPES_DIR,
        help=f"Directory containing recipe .txt/.md files (default: {config.RECIPES_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing Chroma collection and rebuild from scratch.",
    )
    args = parser.parse_args()
    run_ingestion(recipes_dir=args.recipes_dir, reset=args.reset)
