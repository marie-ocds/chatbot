"""
Centralized configuration for the Doctor Dolittle RAG chatbot.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "book" / "the-story-of-doctor-dolittle.pdf"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# Book metadata
BOOK_TITLE = "The Story of Doctor Dolittle"

# LLM Configuration
LLM_MODEL = "gpt-4o-mini-2024-07-18"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Text splitting configuration
SCENE_CHUNK_SIZE = 600
SCENE_CHUNK_OVERLAP = 80

# Retrieval configuration
SCENE_SIMILARITY_TOP_K = 5

# Collection names for ChromaDB
COLLECTION_BOOK = "doctor_dolittle_summary"
COLLECTION_CHAPTERS = "doctor_dolittle_chapters"
COLLECTION_SCENES = "doctor_dolittle_scenes"

# Router prompt template
ROUTER_PROMPT_TEMPLATE = """
You are a query router for a novel question-answering system.

Decide which index should be used to answer the user query.

Routes:
- "book": overall summary, themes, or high-level analysis of the entire book
- "chapter": summaries or overviews of a specific chapter
- "scene": specific events, actions, or factual questions

Rules:
- Use "book" if the query refers to the whole story or book
- Use "chapter" if the query asks about a chapter
- Use "scene" for detailed questions about events

If the query explicitly mentions a chapter number, extract it.
Otherwise return null for chapter_index.

Return ONLY valid JSON matching this schema:
{{
  "route": "book" | "chapter" | "scene",
  "chapter_index": number | null
}}

User query:
{query}
"""