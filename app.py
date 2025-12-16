# app.py

import streamlit as st
import os
import sys
from typing import List  # CRITICAL FIX: Import List
from langchain_core.documents import Document  # CRITICAL FIX: Import Document

# Ensure the backend modules are accessible
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.config import RAGConfig
from backend.rag_single_agent import single_agent_answer_question
from backend.rag_multiagent import multi_agent_answer_question_public  # Use the public alias

# --- Configuration & Initialization ---
CONFIG = RAGConfig()

st.set_page_config(layout="wide")


def display_document_metadata(docs: List[Document]):
    """Displays retrieved document metadata in an expandable section."""
    if not docs:
        st.info("No documents were retrieved for this query.")
        return

    st.subheader(f"Retrieved Documents ({len(docs)} Unique)")

    # Simple list of unique docs based on metadata
    unique_sources = {}
    for doc in docs:
        key = (doc.metadata.get('country'), doc.metadata.get('legal_area'), doc.metadata.get('source_file_name'))
        if key not in unique_sources:
            unique_sources[key] = doc

    for i, doc in enumerate(unique_sources.values(), 1):
        country = doc.metadata.get("country", "N/A")
        area = doc.metadata.get("legal_area", "N/A")
        doc_type = doc.metadata.get("doc_type", "N/A")
        source = doc.metadata.get("source_file_name", "N/A")

        with st.expander(f"**DOC {i}:** {country} / {area} / {source}"):
            st.markdown(f"**Type:** {doc_type}")
            st.text_area(f"Content Snippet", doc.page_content[:500] + "...", height=150, disabled=True)


# --- Streamlit UI Components ---
st.title("⚖️ Multinational Civil Law RAG Agent")
st.caption("Final Project: Comparing Single-Agent ReAct Routing (Task A) vs. Supervisor Multi-Agent (Task B)")

# Sidebar for Configuration
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox(
    "Select Architecture Mode",
    ["Single-Agent (Task A: ReAct Routing)", "Multi-Agent (Task B: Supervisor)"]
)
show_reasoning = st.sidebar.checkbox("Show ReAct/Supervisor Reasoning Trace", value=True)

# Update RAGConfig based on UI selection
if mode == "Multi-Agent (Task B: Supervisor)":
    CONFIG.agentic_mode = "react"  # Both agents use ReAct-style steps
    CONFIG.use_multiagent = True
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Pipeline:** Multi-Agent Supervisor (Task B)")
else:
    CONFIG.agentic_mode = "react"
    CONFIG.use_multiagent = False
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Pipeline:** Single-Agent ReAct (Task A)")

user_query = st.text_area("Enter your Multinational Legal Query:",
                          placeholder="Example: Compare the civil code rules for spousal inheritance in Italy and Slovenia.")

if st.button("Generate Legal Answer"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner(f"Running {mode}..."):
            try:
                # --- Execution ---
                if CONFIG.use_multiagent:
                    # Multi-Agent Pipeline
                    answer, docs, trace = multi_agent_answer_question_public(
                        question=user_query,
                        config=CONFIG,
                        show_reasoning=show_reasoning
                    )
                else:
                    # Single-Agent Pipeline
                    answer, docs, trace = single_agent_answer_question(
                        question=user_query,
                        config=CONFIG,
                        show_reasoning=show_reasoning
                    )

                # --- Display Results ---
                st.header("Final Agent Answer")
                st.markdown(answer)

                st.divider()

                # Display Documents
                display_document_metadata(docs)

                # Display Reasoning Trace
                if show_reasoning and trace:
                    st.header("Agent Execution Trace (Thought, Action, Observation)")
                    st.markdown(trace)

            except Exception as e:
                st.error(f"An unexpected error occurred during pipeline execution: {e}")
                st.exception(e)

st.sidebar.divider()
st.sidebar.markdown(f"**LLM Model:** {CONFIG.llm_model_name}")
st.sidebar.markdown(f"**Embedding Model:** {CONFIG.embedding_model_name}")