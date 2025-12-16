import os
import sys
from typing import List, Dict, Any

# Ensure project backend is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import the necessary components (adjust these imports if needed based on your file structure)
from backend.rag_single_agent import single_agent_answer_question
from backend.rag_multiagent import multi_agent_answer_question_public
from backend.config import RAGConfig

# --- IMPORTANT: SET YOUR KEY HERE ---
# Use the same key you used in backend/config.py
API_KEY = ""
# ------------------------------------

# Define the RAGAS test set (Ground Truth answers are simplified here)
TEST_SET = [
    {
        "question": "What are the mandatory legal requirements for a will to be valid in Estonia?",
        "ground_truth": "A will in Estonia must be written, signed by the testator, and witnessed by two competent persons who sign the will."
    },
    {
        "question": "Compare the primary inheritance code rules between Italy and Slovenia.",
        "ground_truth": "Italian law focuses on the 'reserved portion' (legittima) for family members; Slovenian law includes specific provisions regarding social security aid affecting the estate and procedures for heirless estates."
    }
    # Add all 5 questions here
]


def evaluate_system(architecture_name: str, questions: List[Dict[str, str]]):
    """
    Simulates the RAGAS evaluation run by executing queries and collecting outputs.
    NOTE: You must integrate your actual RAGAS calculation logic here.
    """
    config = RAGConfig()

    if architecture_name == "single_agent":
        config.use_multiagent = False
        run_func = single_agent_answer_question
    else:
        config.use_multiagent = True
        run_func = multi_agent_answer_question_public

    print(f"\n--- Running Evaluation for: {architecture_name} ---")

    results = []

    # --- Integration Point: RAGAS Logic ---
    # 1. Loop through each question.
    # 2. Call your agent function (run_func) to get the 'answer' and 'docs'.
    # 3. Use the RAGAS library to compute metrics based on 'question', 'ground_truth', 'answer', and 'docs'.

    # Example simulation (REPLACE THIS WITH YOUR ACTUAL RAGAS CODE):
    for test in questions:
        print(f"\nProcessing: {test['question']}")

        # 1. Execute Agent
        # In a real setup, you would capture the output and input it into RAGAS
        answer, docs, trace = run_func(test['question'], config, show_reasoning=False)

        # 2. Run RAGAS (SIMULATED STEP)
        # Placeholder for RAGAS computation (e.g., RAGAS(answer, docs, ground_truth).score())
        simulated_scores = {
            "context_precision": 0.9 + len(docs) * 0.01,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
        }

        results.append({
            "question": test['question'],
            "answer": answer,
            "scores": simulated_scores
        })
        print(f"Scores (Simulated): {simulated_scores}")

    return results


def run_full_ragas_evaluation():
    """Main function to run both architectures."""

    if not API_KEY:
        print("ERROR: API_KEY is not set in run_evaluation.py. Please set it to proceed.")
        return

    # Set the key into the environment BEFORE running evaluation functions
    os.environ['OPENAI_API_KEY'] = API_KEY
    print("API Key successfully set in environment for RAGAS evaluation.")

    # --- You would typically run RAGAS here ---
    # Example: Run Single-Agent
    # single_agent_results = evaluate_system("single_agent", TEST_SET)

    # Example: Run Multi-Agent
    # multi_agent_results = evaluate_system("multi_agent", TEST_SET)

    print("\n------------------------------------------------------------")
    print("NOTE: Since RAGAS code is not provided, this script only simulates execution.")
    print("Please integrate your RAGAS evaluation logic into the 'evaluate_system' function.")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    # You would execute this file: python run_evaluation.py
    run_full_ragas_evaluation()