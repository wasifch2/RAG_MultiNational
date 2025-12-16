from __future__ import annotations

import json
from typing import List, Tuple, Optional, Dict, Any
from copy import deepcopy  # Not strictly needed here, but often used for config isolation

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend

# Import retrieval/context helpers from the single-agent file
from .rag_single_agent import _retrieve_with_filters, _build_context, _similarity_rank_and_filter
from .rag_utils import (
    _build_agent_config_log,
)


# =====================================================================
# AGENT CONFIGURATION (Specialized/Virtual Agents)
# =====================================================================
def get_specialized_agent_configs() -> Dict[str, Dict[str, Any]]:
    """Defines the available specialized agents and their fixed filter sets.
    These agents are 'virtual' and operate by applying filters to the single multinational index."""
    return {
        "ItalianDivorceCodeAgent": {
            "filters": [{"country": "Italy", "legal_area": "Divorce", "document_type": "Code"}],
            "description": "Expert in Italian Civil Code provisions regarding separation and divorce (Code/Italy/Divorce)."
        },
        "EstonianInheritanceCaseAgent": {
            "filters": [{"country": "Estonia", "legal_area": "Inheritance", "document_type": "Case"}],
            "description": "Specialized in past legal cases related to inheritance disputes in Estonia (Case/Estonia/Inheritance)."
        },
        "SloveniaAllDivorceAgent": {
            "filters": [{"country": "Slovenia", "legal_area": "Divorce", "document_type": "All"}],
            "description": "General expert on Slovenian divorce law (codes and cases: All/Slovenia/Divorce)."
        },
        "CrossCountryInheritanceCodeComparer": {
            "filters": [{"country": "All", "legal_area": "Inheritance", "document_type": "Code"}],
            "description": "Designed for comparing inheritance civil code provisions across Italy, Estonia, and Slovenia (Code/All/Inheritance)."
        },
        "AllPastCasesTrier": {
            "filters": [{"country": "All", "legal_area": "All", "document_type": "Case"}],
            "description": "Searches all past legal cases in all countries for analogies or trends (Case/All/All)."
        }
    }


# =====================================================================
# SUPERVISOR ROUTING (Thought/Action Step)
# =====================================================================
def _supervisor_route_query(
        question: str,
        llm_backend: LLMBackend,
) -> Tuple[List[str], str]:
    """Uses the Supervisor LLM to select which specialized agents to activate."""

    agent_configs = get_specialized_agent_configs()
    available_agents = "\n".join([
        f"- {name}: {config['description']}"
        for name, config in agent_configs.items()
    ])

    system_prompt = (
        "You are a Supervising Agent. Your sole task is to determine which specialized agents "
        "should be activated to answer the user's complex, multinational legal question. "
        "You must output a JSON list of the agent names you select. DO NOT add any prose or explanation."
    )

    user_prompt = f"""
    Available Specialized Agents and their expertise:
    {available_agents}

    USER QUESTION: {question}

    Output format MUST be a JSON list of agent names (strings) to activate. 
    Example: ["ItalianDivorceCodeAgent", "CrossCountryInheritanceCodeComparer"]

    JSON List Output ONLY:
    """

    try:
        raw_resp = llm_backend.chat(system_prompt, user_prompt)

        # Parse the JSON response, handling LLM formatting
        if '```json' in raw_resp:
            json_str = raw_resp.split('```json')[1].split('```')[0].strip()
        else:
            json_str = raw_resp.strip()

        selected_agents = json.loads(json_str)

        if not isinstance(selected_agents, list):
            raise ValueError("JSON response was not a list.")

        # Validate that the selected names exist
        valid_agents = [name for name in selected_agents if name in agent_configs]

        log = f"Supervisor selected agents (JSON):\n{json.dumps(selected_agents, indent=2)}\n"
        log += f"Valid agents activated: {', '.join(valid_agents)}"

        return valid_agents, log

    except Exception as e:
        log = f"Error in supervisor routing JSON parsing/generation: {e}. Defaulting to no agents."
        return [], log


# =====================================================================
# SUPERVISOR SYNTHESIS (Observation/Answer Step)
# =====================================================================
def _supervisor_synthesize_answers(
        question: str,
        partial_answers: Dict[str, str],
        retrieved_docs: List[Document],
        llm_backend: LLMBackend,
) -> str:
    """Aggregates and synthesizes information from all retrieved contexts and partial answers."""

    # 1. Combine all contexts and partial answers
    combined_context = _build_context(retrieved_docs)

    # 2. Add partial answers to the synthesis prompt
    partial_answers_text = ""
    for agent_name, answer_snippet in partial_answers.items():
        partial_answers_text += f"\n--- Output from {agent_name} ---\n{answer_snippet}\n"

    system_prompt = (
        "You are a sophisticated Legal Supervisor Agent. Your task is to combine "
        "the provided retrieved context and the partial answers from specialized agents "
        "to generate a single, coherent, and comprehensive final answer to the user's question. "
        "Highlight any divergences or comparisons between jurisdictions (Italy, Estonia, Slovenia) "
        "if they are present in the sources. The final answer MUST be consistent with the sources and "
        "clearly traceable back to the context provided. Do not use the partial answers as the final response; "
        "use them, along with the detailed context, to form a better, synthesized response."
    )

    user_prompt = f"""
    USER QUESTION: {question}

    RETRIEVED CONTEXT (Primary Source of Truth):
    {combined_context}

    PARTIAL ANSWERS (For reference and perspective synthesis):
    {partial_answers_text}

    Generate the final, single, synthesized answer below. Ensure it is clear and consistent, 
    and mention the source documents (country/article) where possible.
    """

    # Final LLM call for synthesis
    final_answer = llm_backend.chat(system_prompt, user_prompt)
    return final_answer


# =====================================================================
# MULTI-AGENT CORE
# =====================================================================
def multi_agent_answer_question(
        question: str,
        config: RAGConfig,
        show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Core pipeline for the Multi-Agent Supervisor architecture.
    """
    llm_backend = LLMBackend(config)
    agent_configs = get_specialized_agent_configs()

    # --- Thought/Action (Routing) ---
    selected_agents, routing_log = _supervisor_route_query(question, llm_backend)

    all_retrieved_docs: List[Document] = []
    partial_answers: Dict[str, str] = {}

    # --- Specialized Agent Execution ---
    for agent_name in selected_agents:
        agent_conf = agent_configs[agent_name]
        filters = agent_conf["filters"]

        # 1. Specialized Agent Retrieval (reusing _retrieve_with_filters)
        embedding_model = get_embedding_model(config)

        # Call the proven filtered retrieval function
        docs, retrieval_log_agent = _retrieve_with_filters(
            question=question,
            config=config,
            embedding_model=embedding_model,
            filters=filters,
        )

        all_retrieved_docs.extend(docs)

        # 2. Specialized Agent Answer Generation (Simplified partial answer for synthesis)
        if docs:
            context_agent = _build_context(docs)

            # Simple synthesis prompt for the specialized agent
            system_prompt_partial = (
                f"You are the {agent_name}. Based ONLY on the following context, provide a brief, factual summary "
                f"of the answer to the question. Do not speculate or use external knowledge. "
                f"Context: {context_agent}"
            )
            partial_answer = llm_backend.chat(system_prompt_partial, f"Question: {question}")
            partial_answers[agent_name] = partial_answer
        else:
            partial_answers[agent_name] = "No relevant documents were found by this agent."

    # Remove duplicates from the combined retrieval set before synthesis
    seen_hashes = set()
    unique_docs: List[Document] = []
    for doc in all_retrieved_docs:
        doc_hash = (doc.page_content, doc.metadata.get('source_file_name'), doc.metadata.get('country'))
        if doc_hash not in seen_hashes:
            unique_docs.append(doc)
            seen_hashes.add(doc_hash)

    # --- Answer (Synthesis) ---
    final_answer = _supervisor_synthesize_answers(
        question,
        partial_answers,
        unique_docs,
        llm_backend
    )

    # --- Observation (Trace/Logging) ---
    reasoning_trace: Optional[str] = None
    if show_reasoning:
        # Build the final trace block
        agent_config_log = _build_agent_config_log(config=config, db_map={}, db_descriptions={})

        selected_agent_names = ", ".join(selected_agents) if selected_agents else "None"

        reasoning_trace = (
            f"**Supervisor Thought/Action (Routing)**:\n"
            f"The Supervisor analyzed the query and routed it to the following specialized agents: `{selected_agent_names}`.\n"
            f"Routing Log:\n```text\n{routing_log}\n```\n\n"
            f"**Agent Execution & Partial Answers**:\n"
            f"The specialized agents performed filtered retrieval and returned {len(unique_docs)} unique documents.\n"
            f"Partial Answer Summaries: {json.dumps(partial_answers, indent=2)}\n\n"
            f"**Final Synthesis**:\n"
            f"The Supervisor combined the partial answers and all retrieved context into the final answer.\n\n"
            f"**Agent Configuration**:\n"
            f"```text\n{agent_config_log}\n```"
        )

    return final_answer, unique_docs, reasoning_trace


# Public alias (consistent with single_agent)
def multi_agent_answer_question_public(
        question: str,
        config: RAGConfig,
        show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return multi_agent_answer_question(question, config, show_reasoning)