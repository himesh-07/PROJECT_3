"""
web_app.py
──────────
Streamlit web UI for ChefMind AI.
Runs on http://localhost:8501

Usage:
    streamlit run web_app.py
"""

from __future__ import annotations

import sys
import os

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChefMind AI",
    page_icon="🍴",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍴 ChefMind AI")
st.caption("Culinary RAG agent — adapts recipes to your diet, ingredients & time limits")
st.divider()

# ── Sidebar — backend info ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    try:
        import config
        backend = config.LLM_BACKEND.upper()
        model = (
            config.GEMINI_MODEL   if backend == "GEMINI"  else
            config.OLLAMA_MODEL   if backend == "OLLAMA"  else
            config.WATSONX_MODEL_ID
        )
        st.success(f"**Backend:** {backend}")
        st.info(f"**Model:** `{model}`")
        st.success("**Vector Store:** ChromaDB ✅")
        st.info("**Embeddings:** HuggingFace all-MiniLM-L6-v2" if config.USE_LOCAL
                else "**Embeddings:** IBM slate-125m")
    except Exception as e:
        st.error(f"Config error: {e}")

    st.divider()
    st.markdown(
        "**To change backend**, edit `.env`:\n"
        "```\nLLM_BACKEND=gemini   # or ollama / watsonx\n```"
    )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("chefmind_form"):
    question = st.text_area(
        "🍽️ What would you like to cook or adapt?",
        placeholder="e.g. Make the chocolate chip cookies keto and sugar-free",
        height=80,
    )

    col1, col2 = st.columns(2)

    with col1:
        ingredients = st.text_area(
            "🛒 Available Ingredients",
            placeholder="e.g. almond flour, eggs, butter, erythritol",
            height=100,
            value="not specified",
        )

    with col2:
        restrictions = st.text_area(
            "🚫 Dietary Restrictions / Allergens",
            placeholder="e.g. keto, gluten-free, nut-free",
            height=100,
            value="none",
        )

    constraints = st.text_input(
        "⏱️ Time & Equipment Constraints",
        placeholder="e.g. 30 minutes, standard oven only",
        value="no specific constraints",
    )

    submitted = st.form_submit_button("🔍 Adapt Recipe", use_container_width=True, type="primary")

# ── Pipeline execution ────────────────────────────────────────────────────────
if submitted:
    if not question.strip():
        st.warning("Please enter a cooking goal before submitting.")
        st.stop()

    # Check Chroma index exists
    try:
        import config as _cfg
        if not os.path.exists(_cfg.CHROMA_PERSIST_DIR):
            st.error(
                "⚠️ Recipe index not found. "
                "Run **`python ingest.py`** in your terminal first, then refresh."
            )
            st.stop()
    except Exception:
        pass

    with st.spinner("🔍 Retrieving recipe context from ChromaDB…"):
        try:
            import chefmind
        except Exception as e:
            st.error(f"Import error: {e}")
            st.stop()

    with st.spinner(f"🤖 Generating recipe adaptation…"):
        try:
            answer = chefmind.run_chefmind(
                question=question,
                available_ingredients=ingredients,
                dietary_restrictions=restrictions,
                constraints=constraints,
            )
        except Exception as e:
            st.error(f"**Error during generation:** {e}")
            st.stop()

    # ── Output ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Recipe Adaptation")
    st.markdown(answer)

    # Download button
    st.download_button(
        label="⬇️ Download as Markdown",
        data=answer,
        file_name="chefmind_recipe.md",
        mime="text/markdown",
    )
