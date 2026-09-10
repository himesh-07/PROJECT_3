"""
langflow_component.py
─────────────────────
IBM Langflow integration for ChefMind AI.

This file defines a Langflow CustomComponent that wraps the existing
ChefMind RAG pipeline so it can be dropped into any Langflow flow
as a visual node.

HOW TO USE IN LANGFLOW
──────────────────────
1. Start Langflow:
       langflow run
   or via Docker:
       docker run -p 7860:7860 langflowai/langflow:latest

2. In the Langflow UI, go to Custom Components → Upload Component.
   Upload THIS file (langflow_component.py).

3. The "ChefMind AI (watsonx RAG)" node will appear in your component
   palette. Drag it onto the canvas.

4. Wire it:
   - Input: connect a TextInput node to each of the four input ports
     (question, available_ingredients, dietary_restrictions, constraints).
   - Output: connect the output port to a TextOutput or ChatOutput node.

5. Set the IBM watsonx credentials in the component's configuration panel
   OR ensure the .env file is present in the working directory before
   starting Langflow.

EXISTING PIPELINE — UNCHANGED
──────────────────────────────
This component calls chefmind.run_chefmind() which in turn uses:
  - retriever.py  → ChromaDB semantic search (IBM slate embeddings)
  - prompt_builder.py → ChefMind system prompt injection
  - IBM Granite via ibm-watsonx-ai SDK (chat API)

Nothing in the existing pipeline is modified.
"""

from __future__ import annotations

# Langflow imports — available once `langflow>=1.1.0` is installed.
from langflow.custom import CustomComponent
from langflow.schema import Data


class ChefMindRAGComponent(CustomComponent):
    """
    Langflow node that runs the full ChefMind AI RAG pipeline.

    Inputs
    ──────
    question              : str  — cooking goal / recipe adaptation request
    available_ingredients : str  — comma-separated pantry items
    dietary_restrictions  : str  — allergens / diet flags (default: "none")
    constraints           : str  — time, equipment limits (default: "none")

    Output
    ──────
    Data  — Langflow Data object carrying the formatted recipe markdown
            in the `.text` field.
    """

    # ── Langflow node metadata ────────────────────────────────────────────────
    display_name: str = "ChefMind AI (watsonx RAG)"
    description: str = (
        "Adapts recipes from your IBM watsonx-powered culinary knowledge base. "
        "Uses ChromaDB + IBM Granite + RAG grounding."
    )
    icon: str = "🍴"
    name: str = "ChefMindRAGComponent"

    # ── Input field declarations ──────────────────────────────────────────────
    inputs = [
        {
            "name": "question",
            "display_name": "Cooking Goal",
            "type": "str",
            "required": True,
            "placeholder": "e.g. Make the chocolate chip cookies keto and sugar-free",
            "info": "What the user wants to cook or adapt.",
        },
        {
            "name": "available_ingredients",
            "display_name": "Available Ingredients",
            "type": "str",
            "required": False,
            "value": "not specified",
            "info": "Comma-separated list of ingredients the user has on hand.",
        },
        {
            "name": "dietary_restrictions",
            "display_name": "Dietary Restrictions / Allergens",
            "type": "str",
            "required": False,
            "value": "none",
            "info": "e.g. keto, gluten-free, nut-free, vegan",
        },
        {
            "name": "constraints",
            "display_name": "Time & Equipment Constraints",
            "type": "str",
            "required": False,
            "value": "no specific constraints",
            "info": "e.g. 30 minutes, standard oven only, no stand mixer",
        },
    ]

    # ── Output field declaration ──────────────────────────────────────────────
    outputs = [
        {
            "name": "recipe_adaptation",
            "display_name": "Recipe Adaptation",
            "method": "build",
        }
    ]

    # ── Core logic ────────────────────────────────────────────────────────────
    def build(
        self,
        question: str,
        available_ingredients: str = "not specified",
        dietary_restrictions: str = "none",
        constraints: str = "no specific constraints",
    ) -> Data:
        """
        Invokes the ChefMind RAG pipeline and returns a Langflow Data object.

        The existing chefmind.run_chefmind() function is called unchanged.
        ChromaDB, IBM embeddings, and IBM Granite are all invoked internally.
        """
        # Lazy import so this file can be syntax-checked without the full
        # pipeline environment being present (e.g., during Langflow component upload).
        import chefmind  # noqa: PLC0415

        answer = chefmind.run_chefmind(
            question=question,
            available_ingredients=available_ingredients,
            dietary_restrictions=dietary_restrictions,
            constraints=constraints,
        )

        # Wrap in Langflow's Data schema — .text is the standard content field.
        return Data(text=answer)
