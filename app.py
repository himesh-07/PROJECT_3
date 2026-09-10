"""
app.py
──────
Interactive CLI entrypoint for ChefMind AI.

Usage
─────
    python app.py                           # interactive mode (prompts for input)
    python app.py --question "Make a keto chocolate cake" \
                  --ingredients "almond flour, eggs, butter, cocoa powder" \
                  --restrictions "keto, sugar-free" \
                  --constraints "45 minutes, standard oven"
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

import chefmind

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold white]🍴  ChefMind AI[/bold white]\n"
            "[dim]Culinary RAG agent powered by IBM watsonx[/dim]",
            border_style="blue",
            padding=(1, 4),
        )
    )


def _interactive_mode() -> dict[str, str]:
    """Collect inputs interactively from the user."""
    console.print("\n[bold]Answer the prompts below.[/bold] "
                  "Press [dim]Enter[/dim] to accept defaults.\n")

    question = Prompt.ask(
        "[cyan]What would you like to cook or adapt?[/cyan]"
    )
    ingredients = Prompt.ask(
        "[cyan]Available ingredients[/cyan]",
        default="not specified",
    )
    restrictions = Prompt.ask(
        "[cyan]Dietary restrictions / allergens[/cyan]",
        default="none",
    )
    constraints = Prompt.ask(
        "[cyan]Time & equipment constraints[/cyan]",
        default="no specific constraints",
    )
    return {
        "question": question,
        "available_ingredients": ingredients,
        "dietary_restrictions": restrictions,
        "constraints": constraints,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chefmind",
        description="ChefMind AI — IBM watsonx culinary RAG agent",
    )
    parser.add_argument(
        "--question", "-q",
        help="Cooking goal or recipe adaptation request.",
    )
    parser.add_argument(
        "--ingredients", "-i",
        default="not specified",
        help="Comma-separated list of available ingredients.",
    )
    parser.add_argument(
        "--restrictions", "-r",
        default="none",
        help="Dietary restrictions or allergens (e.g., 'keto, nut-free').",
    )
    parser.add_argument(
        "--constraints", "-c",
        default="no specific constraints",
        help="Time, equipment, or skill-level limits.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner()

    args = _parse_args()

    # If --question was not supplied, fall back to interactive mode
    if args.question:
        inputs = {
            "question": args.question,
            "available_ingredients": args.ingredients,
            "dietary_restrictions": args.restrictions,
            "constraints": args.constraints,
        }
    else:
        inputs = _interactive_mode()

    # Validate that a question was provided
    if not inputs["question"].strip():
        console.print("[red]✖  A cooking question is required. Exiting.[/red]")
        sys.exit(1)

    # Run the full RAG pipeline
    console.print()
    answer = chefmind.run_chefmind(**inputs)

    # Render the markdown response in the terminal
    console.print()
    console.rule("[bold blue]ChefMind AI — Recipe Adaptation[/bold blue]")
    console.print(Markdown(answer))
    console.rule()


if __name__ == "__main__":
    main()
