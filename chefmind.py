"""
chefmind.py
───────────
Core pipeline orchestrator for ChefMind AI.

Wires together:
  retriever      →  semantic search over the recipe knowledge base
  prompt_builder →  injects retrieved context + user constraints
  LLM backend    →  gemini | ollama | watsonx
                    controlled by LLM_BACKEND in .env

The single public function `run_chefmind()` accepts the five user fields
and returns the model's formatted recipe adaptation as a string.
"""

from __future__ import annotations

from rich.console import Console

import config
import prompt_builder
import retriever

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_gemini(prompt: str) -> str:
    """Call Google Gemini API (LLM_BACKEND=gemini)."""
    import google.generativeai as genai  # noqa: PLC0415

    console.print(
        f"[cyan]🤖  Sending prompt to [bold]Gemini {config.GEMINI_MODEL}[/bold]…[/cyan]"
    )
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def _generate_local(prompt: str) -> str:
    """Call the locally running Ollama model (LLM_BACKEND=ollama)."""
    import ollama  # noqa: PLC0415

    console.print(
        f"[cyan]🤖  Running [bold]{config.OLLAMA_MODEL}[/bold] locally via Ollama…[/cyan]"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _generate_watsonx(prompt: str) -> str:
    """Call IBM Granite via the watsonx.ai SDK (LLM_BACKEND=watsonx)."""
    from ibm_watsonx_ai import Credentials  # noqa: PLC0415
    from ibm_watsonx_ai.foundation_models import ModelInference  # noqa: PLC0415

    console.print(
        f"[cyan]🤖  Sending prompt to IBM [bold]{config.WATSONX_MODEL_ID}[/bold]…[/cyan]"
    )
    model = ModelInference(
        model_id=config.WATSONX_MODEL_ID,
        credentials=Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY),
        project_id=config.WATSONX_PROJECT_ID,
        params=config.GEN_PARAMS,
    )
    response = model.chat(messages=[{"role": "user", "content": prompt}])
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_chefmind(
    question: str,
    available_ingredients: str = "not specified",
    dietary_restrictions: str = "none",
    constraints: str = "no specific constraints",
) -> str:
    """
    Full RAG pipeline: retrieve → build prompt → generate → return.

    Parameters
    ──────────
    question              What the user wants to cook / the adaptation goal.
    available_ingredients Comma-separated ingredients the user already has.
    dietary_restrictions  Dietary flags or allergens (e.g., "keto, nut-free").
    constraints           Time, equipment, or skill-level limits.

    Returns
    ───────
    The model's formatted ChefMind recipe response (markdown string).
    """
    # ── Step 1: Retrieve relevant recipe context ──────────────────────────────
    console.print("[cyan]🔍  Retrieving recipe context…[/cyan]")
    context = retriever.retrieve_context(question)

    # ── Step 2: Build the fully injected prompt ───────────────────────────────
    prompt = prompt_builder.build_prompt(
        context=context,
        question=question,
        available_ingredients=available_ingredients,
        dietary_restrictions=dietary_restrictions,
        constraints=constraints,
    )

    # ── Step 3: Route to correct LLM backend ─────────────────────────────────
    backend = config.LLM_BACKEND
    if backend == "gemini":
        return _generate_gemini(prompt)
    elif backend == "ollama":
        return _generate_local(prompt)
    else:
        return _generate_watsonx(prompt)
