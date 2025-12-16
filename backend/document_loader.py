import json
import os
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

# --- Configuration (Adjust this path if necessary) ---
# Assuming 'Contest_Data' is in the main 'Rag' folder, which is sibling to 'RAG_4_Scratch-main'
# If DATA_ROOT is inside RAG_4_Scratch-main, adjust accordingly.
# Based on your path: C:/Users/DELL/Desktop/Rag/RAG_4_Scratch-main/Contest_Data
# Since this script is in 'backend', two steps up takes us to 'RAG_4_Scratch-main'
DATA_ROOT = 'C:/Users/DELL/Desktop/Rag/RAG_4_Scratch-main/Contest_Data'

# Define the keywords to categorize the data for automated tagging
COUNTRIES = ['italy', 'estonia', 'slovenia']
LEGAL_AREAS = ['inheritance', 'divorce']
# Using keywords that match your folder names like 'Estonian_cases_json_processed'
CASE_KEYWORDS = ['case', 'cases', 'processed']


# --- Utility Function to Extract Content and Normalize Metadata ---
# Updated to accept additional_metadata
def _extract_docs_from_json_object(obj: Any, source: str, additional_metadata: dict) -> List[Document]:
    """
    Accepts:
      - A dict with 'content'/'text'/'corpus' and optional 'metadata'
      - A list of such dicts
    Returns a list of LangChain Document objects with added metadata.
    """

    docs: List[Document] = []

    def normalize_single(item: dict) -> Document | None:
        content_key = None
        for k in ["content", "text", "corpus", "Text"]:
            if k in item:
                content_key = k
                break

        if content_key is None:
            return None

        content = item[content_key]
        if not isinstance(content, str):
            content = str(content)

        # Merge existing 'metadata' field from JSON with automatically generated tags
        meta = item.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {"metadata_raw": str(meta)}

        # Combine all metadata: auto-tags (additional_metadata) are injected last
        final_meta = {
            "source": source,
            **meta,
            **additional_metadata
        }

        # Ensure the content is not empty
        if not content.strip():
            return None

        return Document(page_content=content, metadata=final_meta)

    # Handle lists of documents or a single document object
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                d = normalize_single(item)
                if d is not None:
                    docs.append(d)
    elif isinstance(obj, dict):
        d = normalize_single(obj)
        if d is not None:
            docs.append(d)

    return docs


# --- Main Loading Function (Replaces load_documents_from_folders) ---
def load_all_multinational_documents() -> List[Document]:
    """
    Loads all documents from DATA_ROOT recursively and tags them with
    country, legal_area, and document_type based on the path.
    """
    data_root_path = Path(DATA_ROOT)
    if not data_root_path.exists():
        print(f"[document_loader] Error: Data root path not found at {data_root_path}")
        return []

    print(f"Starting multinational document loading from: {data_root_path}")
    all_docs: List[Document] = []

    # Use rglob to find all .json files recursively
    for json_file in data_root_path.rglob("*.json"):
        # Convert path to string for keyword searching
        path_str = str(json_file.as_posix()).lower()

        # 1. Automatic Metadata Detection from Folder Structure
        # The logic finds the country/area that appears in the path string.
        country = next((c.capitalize() for c in COUNTRIES if c in path_str), None)
        legal_area = next((a.capitalize() for a in LEGAL_AREAS if a in path_str), None)

        # Determine Document Type (Code or Case)
        is_case_folder = any(keyword in path_str for keyword in CASE_KEYWORDS)
        doc_type = 'Case' if is_case_folder else 'Code'

        # Must have both country and legal area for routing
        if country is None or legal_area is None:
            continue

            # Metadata dictionary to inject into the LangChain Document
        injected_metadata = {
            'country': country,
            'legal_area': legal_area,
            'document_type': doc_type,
            'source_file_name': json_file.name
        }

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Pass the injected_metadata to the extraction helper
            docs = _extract_docs_from_json_object(
                data,
                source=str(json_file.as_posix()),
                additional_metadata=injected_metadata
            )
            all_docs.extend(docs)

        except Exception as e:
            print(f"[document_loader] Error loading {json_file}: {e}")

    print(f"Successfully loaded {len(all_docs)} documents with routing metadata.")
    return all_docs


# We keep this for compatibility if other scripts still call it, but it's now deprecated
# It is recommended to update all callers to use load_all_multinational_documents()
def load_documents_from_folders(folders: List[str]) -> List[Document]:
    print(
        "[document_loader] Warning: 'load_documents_from_folders' is deprecated. Use 'load_all_multinational_documents'.")
    return load_all_multinational_documents()


# --- For quick testing and verification ---
if __name__ == '__main__':
    documents = load_all_multinational_documents()

    # Print a sample to verify the metadata is correct
    if documents:
        print("\n--- Sample Document Metadata (Verification) ---")
        for i, doc in enumerate(documents[:5]):
            # Assign the cleaned content snippet to a variable to avoid the SyntaxError
            content_snippet = doc.page_content[:80].replace('\n', ' ').replace('\r', ' ')

            print(f"Document {i + 1}:")
            print(f"  Country: {doc.metadata.get('country')}")
            print(f"  Legal Area: {doc.metadata.get('legal_area')}")
            print(f"  Doc Type: {doc.metadata.get('document_type')}")
            print(f"  Source File: {doc.metadata.get('source_file_name')}")
            print(f"  Content Snippet: {content_snippet}...")
            print("-" * 30)