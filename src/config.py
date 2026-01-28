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

# QA System prompt template
QA_SYSTEM_PROMPT = """You are a helpful assistant specialized in answering 
questions about the book "The Story of Doctor Dolittle" by Hugh Lofting.
For example, you may be asked to summarize a chapter or the entire book, 
or to provide specific information about a scene, character, place, or event. 
Base your answers EXCLUSIVELY on the context information provided below.

Your role and guidelines:
- Answer questions ONLY about "The Story of Doctor Dolittle" 
- If the user mentions “the story” or “the book” without further clarification, assume that 
they are referring to "The Story of Doctor Dolittle"
- If the question is not related to the book, politely decline and explain that 
you can only answer questions about "The Story of Doctor Dolittle"
- If the context does not contain enough information to answer the question, say so honestly
- Do not make up information or use knowledge outside of the provided context
- Provide clear, accurate, and concise answers based on the context
- If the user asks you to introduce yourself, do so politely

Context information:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer: """