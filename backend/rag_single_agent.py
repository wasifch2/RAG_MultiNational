from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# --- CRITICAL IMPORTS ---
from langchain_core.documents import Document 
from .config import RAGConfig 
# ------------------------

from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import get_multinational_vector_store
from .rag_utils import (
    _build_agent_config_log,
)


# =====================================================================
# Context builder + similarity filtering (with logging)
# =====================================================================
def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
    chunks = []
    total = 0
    for i, d in enumerate(docs):
        # Updated metadata extraction to use the new tags
        country = d.metadata.get("country", "Unknown Country")
        area = d.metadata.get("legal_area", "Unknown Area")
        doc_type = d.metadata.get("document_type", "Unknown Type")
        src = d.metadata.get("source_file_name", "unknown")

        # Ensures transparency by including country/area/type in context header
        header = f"[DOC {i + 1} | {country}/{area} | Type: {doc_type} | Source: {src}]\n"
        text = d.page_content
        piece = header + text + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _similarity_rank_and_filter(
        question: str,
        docs: List[Document],
        embedding_model,
        top_k: int,
        min_sim: float = 0.1,
) -> Tuple[List[Document], str]:
    """
    Rank docs by cosine similarity and filter below min_sim.
    Returns (filtered_docs, log_string).
    """
    log_lines: List[str] = []

    if not docs:
        log_lines.append("No documents returned from base retriever.")
        return [], "\n".join(log_lines)

    # Note: Using np.array for numerical operations
    # Ensure get_embedding_model is correctly configured in embeddings.py
    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    doc_texts = [d.page_content for d in docs]
    doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype="float32")

    q_norm = np.linalg.norm(q_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    denom = np.maximum(q_norm * doc_norms, 1e-8)
    sims = (doc_vecs @ q_vec) / denom

    num_raw = len(docs)
    sims_min = float(np.min(sims))
    sims_max = float(np.max(sims))
    sims_mean = float(np.mean(sims))

    indices = [i for i, s in enumerate(sims) if s >= min_sim]
    num_after_threshold = len(indices)

    if not indices:
        log_lines.append(
            f"Similarity filtering: {num_raw} raw docs → 0 kept "
            f"(threshold={min_sim:.3f}, "
            f"sim range=[{sims_min:.3f}, {sims_max:.3f}], mean={sims_mean:.3f})."
        )
        return [], "\n".join(log_lines)

    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    final_docs = [docs[i] for i in indices_sorted]

    sims_kept = sims[indices_sorted]
    sims_kept_min = float(np.min(sims_kept))
    sims_kept_max = float(np.max(sims_kept))
    sims_kept_mean = float(np.mean(sims_kept))

    log_lines.append(
        "Similarity filtering + reranking:\n"
        f"- Raw docs from retriever: {num_raw}\n"
        f"- Docs above threshold {min_sim:.3f}: {num_after_threshold}\n"
        f"- Final top_k={top_k} docs kept: {len(final_docs)}\n"
        f"- Similarity stats (all raw): min={sims_min:.3f}, max={sims_max:.3f}, "
        f"mean={sims_mean:.3f}\n"
        f"- Similarity stats (kept):   min={sims_kept_min:.3f}, max={sims_kept_max:.3f}, "
        f"mean={sims_kept_mean:.3f}"
    )

    return final_docs, "\n".join(log_lines)


# =====================================================================
# AGENTIC ROUTING (Thought/Action Step) - NEW CORE LOGIC
# =====================================================================

def _decide_and_filter_retrieval_params(
        question: str,
        llm_backend: LLMBackend,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Forces the LLM to output a list of metadata filters (routing decision).
    Returns (filter_list, log_string).
    """

    system_prompt = (
        "You are the routing component of a Civil Law RAG Agent. Your sole task "
        "is to analyze the user's question and output a JSON list specifying the "
        "EXACT metadata filters needed for document retrieval from the single multinational database. "
        "Your output MUST be a valid JSON list. DO NOT add any prose, comments, or explanations."
    )

    user_prompt = f"""
    USER QUESTION: {question}

    You must select relevant subsets based on the following available metadata fields:
    - 'country': 'Italy', 'Estonia', or 'Slovenia'. Use 'All' for comparison or multi-country queries.
    - 'legal_area': 'Inheritance' or 'Divorce'. Use 'All' if both are relevant.
    - 'document_type': 'Code' (civil codes) or 'Case' (past cases). Use 'All' if both are relevant.

    If the question is about a specific case type in a specific country, use that filter. 
    If it compares laws across all countries, use country: 'All'.

    Output format MUST be a JSON list of objects, where each object defines a necessary filter set. 
    Example 1 (Specific): [{{
        "country": "Estonia", 
        "legal_area": "Divorce", 
        "document_type": "Case"
    }}]
    Example 2 (Comparison): [{{
        "country": "All", 
        "legal_area": "Inheritance", 
        "document_type": "Code"
    }}]

    JSON List Output ONLY:
    """

    try:
        raw_resp = llm_backend.chat(system_prompt, user_prompt)

        # Attempt to clean and parse the JSON response
        if '```json' in raw_resp:
            json_str = raw_resp.split('```json')[1].split('```')[0].strip()
        else:
            json_str = raw_resp.strip()

        filters = json.loads(json_str)

        log = f"Routing decision (JSON):\n{json.dumps(filters, indent=2)}\n"

        if not isinstance(filters, list):
            raise ValueError("JSON response was not a list.")

        # Basic validation: ensure filters contain required keys
        valid_filters = []
        for f in filters:
            if all(key in f for key in ['country', 'legal_area', 'document_type']):
                valid_filters.append(f)
            else:
                log += f"Warning: Filter skipped due to missing keys: {f}\n"

        return valid_filters, log

    except Exception as e:
        log = f"Error parsing routing JSON: {e}. Defaulting to NO retrieval."
        return [], log


# --- NEW FUNCTION: Retrieval with Metadata Filtering ---
def _retrieve_with_filters(
        question: str,
        config: RAGConfig,
        embedding_model,
        filters: List[Dict[str, str]],
) -> Tuple[List[Document], str]:
    """
    Retrieves documents from the single multinational store using metadata filters.
    Returns (docs_kept, log_string).
    """
    log_lines: List[str] = ["Starting filtered retrieval from multinational_rag_index."]
    all_filtered_docs: List[Document] = []

    # 1. Load the single multinational store
    vector_store = get_multinational_vector_store()
    if vector_store is None:
        log_lines.append("Error: Failed to load multinational vector store.")
        return [], "\n".join(log_lines)

    # 2. Iterate through each filter set (routing decision)
    for i, filter_set in enumerate(filters):
        country = filter_set.get('country')
        legal_area = filter_set.get('legal_area')
        doc_type = filter_set.get('document_type')

        log_lines.append(f"\n- Filter Set {i + 1}: Country={country}, Area={legal_area}, Type={doc_type}")

        # The core of metadata filtering in LangChain (a callable function)
        def metadata_filter(doc_metadata):
            match_country = (country == 'All') or (doc_metadata.get('country') == country)
            match_area = (legal_area == 'All') or (doc_metadata.get('legal_area') == legal_area)
            match_type = (doc_type == 'All') or (doc_metadata.get('document_type') == doc_type)
            return match_country and match_area and match_type

        # Use a large k for base retrieval before similarity filtering/reranking
        k_base = max(config.top_k * 3, config.top_k)

        # Create the retriever with the custom metadata filter
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k_base,
                "lambda_func": metadata_filter
            }
        )

        try:
            raw_docs = retriever.invoke(question)
        except Exception as e:
            log_lines.append(f"  Error during retrieval for filter set {i + 1}: {e}")
            continue

        log_lines.append(f"  Raw docs retrieved for filter set: {len(raw_docs)}")

        # 3. Apply Reranking/Filtering (using your existing utility)
        docs_set, sim_log = _similarity_rank_and_filter(
            question=question,
            docs=raw_docs,
            embedding_model=embedding_model,
            top_k=config.top_k,
            min_sim=0.1,
        )

        log_lines.append(f"  Docs kept after reranking: {len(docs_set)}")
        all_filtered_docs.extend(docs_set)

    # --- DEDUPLICATION FIX ---
    seen_hashes = set()
    unique_docs: List[Document] = []
    for doc in all_filtered_docs:
        # Create a hashable tuple based on content and key metadata fields
        doc_hash = (doc.page_content, doc.metadata.get('source_file_name'), doc.metadata.get('country'))
        if doc_hash not in seen_hashes:
            unique_docs.append(doc)
            seen_hashes.add(doc_hash)

    log_lines.append(f"\nTotal unique documents kept for context: {len(unique_docs)}")

    return unique_docs, "\n".join(log_lines)


# =====================================================================
# Agentic decision: do we need retrieval?
# =====================================================================
# NOTE: This function is simplified and reliable for compliant LLMs (like OpenAI).
def _decide_need_retrieval(
        question: str,
        config: RAGConfig,
        llm_backend: LLMBackend,
) -> Tuple[bool, str]:
    system_prompt = (
        "You are a classifier that decides if a question needs external documents "
        "to answer accurately.\n"
        "Reply with a single word:\n"
        "- 'YES' if external documents or context WOULD help or are needed.\n"
        "- 'NO' if the question can be answered reliably from general knowledge."
    )
    user_prompt = f"Question:\n{question}\n\nAnswer YES or NO only."

    resp = llm_backend.chat(system_prompt, user_prompt).strip().lower()

    if "yes" in resp:
        return True, f"Retrieval decision: Model suggested retrieval ('{resp}') → USE retrieval."

    if "no" in resp and len(resp.split()) <= 2:
        return False, f"Retrieval decision: Model answered '{resp}' → NO retrieval."

    return True, f"Retrieval decision: Ambiguous/non-compliant response ('{resp}') → default to USE retrieval."


# =====================================================================
# Helper: summarized Observation text (using content + LLM)
# =====================================================================
def _build_observation_text(
        question: str,
        need_retrieval: bool,
        docs: List[Document],
        llm_backend: LLMBackend,
        used_filter_names: List[str],  # Adapted to use filter names
) -> str:
    if not need_retrieval:
        return "No external vector database was used; the answer relies on internal knowledge."

    if need_retrieval and not docs:
        if used_filter_names:
            db_list = ", ".join(sorted(set(used_filter_names)))
            return (
                f"Retrieval was attempted using filters targeting: {db_list}, "
                "but no sufficiently relevant documents were found."
            )
        return "Retrieval was attempted, but no sufficiently relevant documents were found."

    doc_blocks = []
    for i, d in enumerate(docs, start=1):
        country = d.metadata.get("country", "Unknown Country")
        area = d.metadata.get("legal_area", "Unknown Area")
        doc_type = d.metadata.get("document_type", "Unknown Type")
        src = d.metadata.get("source_file_name", "unknown")
        snippet = d.page_content[:400].replace("\n", " ").strip()
        doc_blocks.append(
            f"[DOC {i}] Country={country}, Area={area}, Type={doc_type} | source={src}\n"
            f"Snippet: {snippet}\n"
        )

    docs_text = "\n\n".join(doc_blocks)
    filter_list = ", ".join(sorted(set(used_filter_names))) if used_filter_names else "unknown"

    system_prompt = (
        "You are summarizing how retrieved documents from the filtered legal database "
        "help answer a user's question.\n"
        "You MUST be concise and high-level. Do NOT reveal detailed chain-of-thought.\n"
        "Your job:\n"
        "- Mention briefly WHICH legal contexts (e.g., 'Italian Divorce Law') were queried.\n"
        "- In 2–4 bullet points, explain at a high level how the retrieved content "
        "is useful or relevant for answering the question (e.g., 'provided specific articles on succession').\n"
        "- Keep the explanation short."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Contexts targeted: {filter_list}\n\n"
        f"Retrieved documents (summarized):\n{docs_text}\n\n"
        "Now produce a SHORT observation in this format:\n\n"
        "Contexts targeted: <comma-separated list>\n"
        "- bullet point 1\n"
        "- bullet point 2\n"
        "- (optional) bullet point 3\n"
    )

    explanation = llm_backend.chat(system_prompt, user_prompt)
    return explanation


# =====================================================================
# SINGLE-AGENT CORE (ReAct-style) - UPDATED
# =====================================================================
def _single_agent_answer_question_core(
        question: str,
        config: RAGConfig,
        show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    ReAct-style single-agent RAG pipeline adapted for multinational routing.
    """
    llm_backend = LLMBackend(config)

    # ---- Thought: need retrieval? ----
    # Step 1: Decide if external knowledge is needed (YES/NO)
    need_retrieval, decision_log = _decide_need_retrieval(
        question, config, llm_backend
    )

    retrieved_docs: List[Document] = []
    context = ""
    retrieval_log = ""
    filter_set: List[Dict[str, str]] = []
    used_filter_names: List[str] = []

    # ---- Action: if needed, decide filters & retrieve ----
    if need_retrieval:
        # Step 2: Decide Filters (Routing) - This is the core 'Action'
        filter_set, db_selection_log = _decide_and_filter_retrieval_params(
            question=question,
            llm_backend=llm_backend,
        )

        if filter_set:
            # Generate descriptive names for the observation log
            used_filter_names = [
                f"{f['country']} {f['legal_area']} {f['document_type']}"
                for f in filter_set
            ]

            # Step 3: Retrieve with Filters
            embedding_model = get_embedding_model(config)

            retrieved_docs, retrieval_log = _retrieve_with_filters(
                question=question,
                config=config,
                embedding_model=embedding_model,
                filters=filter_set,
            )
            context = _build_context(retrieved_docs)

            # Combine decision log with retrieval log for trace
            retrieval_log = f"{db_selection_log}\n{retrieval_log}"

        else:
            # Model decided retrieval was needed, but returned no valid filters
            need_retrieval = False
            retrieval_log = db_selection_log

    # ---- Answer: main LLM call ----
    if config.agentic_mode == "react":
        system_prompt = (
            "You are an agentic reasoning assistant. "
            "If context from retrieved documents is provided, use it as your "
            "primary source of truth. If no context is provided, rely on your "
            "own knowledge. In all cases, do not reveal your internal chain-of-"
            "thought; provide only a clear final answer. If you are uncertain, "
            "say so explicitly."
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append(
            "Provide a clear, concise final answer without exposing your internal steps."
        )
        user_prompt = "\n\n".join(user_parts)
    else:
        system_prompt = (
            "You are a helpful assistant answering questions. "
            "If context from retrieved documents is provided, treat it as the most "
            "authoritative source. If no context is provided, rely on your own "
            "knowledge to answer. If the question cannot be answered reliably, "
            "explain that you are unsure."
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append("Provide a concise, accurate answer.")
        user_prompt = "\n\n".join(user_parts)

    answer = llm_backend.chat(system_prompt, user_prompt)

    # ---- Optional ReAct-style trace + retrieval + agent config logs ----
    reasoning_trace: Optional[str] = None
    if config.agentic_mode == "react" and show_reasoning:
        # Step 4: Observation (LLM summarizes the process)
        if need_retrieval:
            thought_str = (
                "The agent analyzed the question to understand its topic and "
                "determined that consulting the filtered legal corpus would "
                "improve the answer."
            )
        else:
            thought_str = (
                "The agent analyzed the question and decided it could be answered "
                "reliably without consulting the external database."
            )

        if need_retrieval and filter_set:
            action_str = (
                    "The agent explicitly routed the query to target the following metadata filters: "
                    + ", ".join(f"`{n}`" for n in used_filter_names)
                    + "."
            )
        elif need_retrieval and not filter_set:
            action_str = (
                "The agent considered retrieval but, after analysis, determined no specific "
                "metadata filters could be applied, or the filters were invalid."
            )
        else:
            action_str = (
                "The agent skipped retrieval and relied solely on its own knowledge."
            )

        observation_str = _build_observation_text(
            question=question,
            need_retrieval=need_retrieval,
            docs=retrieved_docs,
            llm_backend=llm_backend,
            used_filter_names=used_filter_names,
        )

        retrieval_log_block = retrieval_log

        # FIX: Pass empty dictionaries for obsolete multi-DB arguments (db_map, db_descriptions)
        agent_config_log = _build_agent_config_log(
            config=config,
            db_map={},  # Pass empty dict
            db_descriptions={},  # Pass empty dict
        )

        reasoning_trace = (
            f"**Thought**: {thought_str}\n\n"
            f"**Action**: {action_str}\n\n"
            f"**Observation**:\n{observation_str}\n\n"
            f"**Retrieval / Filtering Log**:\n"
            f"```text\n{retrieval_log_block}\n```\n\n"
            f"**Agent Configuration**:\n"
            f"```text\n{agent_config_log}\n```"
        )

    return answer, retrieved_docs, reasoning_trace


# Public alias
def single_agent_answer_question(
        question: str,
        config: RAGConfig,
        show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return _single_agent_answer_question_core(question, config, show_reasoning)