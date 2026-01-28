"""
Retrieval module for query routing and answering.
"""
from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Literal
import json
from pydantic import ValidationError

from src.config import ROUTER_PROMPT_TEMPLATE, SCENE_SIMILARITY_TOP_K


class RouteDecision(BaseModel):
    """
    Pydantic model for routing decisions.
    """
    route: Literal["book", "chapter", "scene"]
    chapter_index: Optional[int] = Field(default=None)

ROUTER_PROMPT = PromptTemplate(ROUTER_PROMPT_TEMPLATE)


def route_query(query: str) -> RouteDecision:
    """
    Route a user query to the appropriate index.

    Args:
        query: User's question

    Returns:
        RouteDecision: Decision object with route and optional chapter_index
    """
    llm = Settings.llm

    prompt = ROUTER_PROMPT.format(query=query)
    response = llm.complete(prompt)

    try:
        raw = json.loads(response.text)
        decision = RouteDecision.model_validate(raw)
        return decision

    except (json.JSONDecodeError, ValidationError) as e:
        # Safe fallback: default to scene-level retrieval
        return RouteDecision(route="scene", chapter_index=None)


def chapter_index_engine_filtered(chapter_index, chapter_index_value: int):
    """
    Create a filtered query engine for a specific chapter.

    Args:
        chapter_index: The chapter VectorStoreIndex
        chapter_index_value: The chapter number to filter by

    Returns:
        Query engine filtered by chapter_index
    """
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="chapter_index",
                value=chapter_index_value,
            )
        ]
    )

    return chapter_index.as_query_engine(filters=filters, similarity_top_k=1)


def answer_query(query: str, book_index, chapter_index, scenes_index):
    """
    Answer a user query by routing to the appropriate index.

    Args:
        query: User's question
        book_index: VectorStoreIndex for the global summary
        chapter_index: VectorStoreIndex for chapters
        scenes_index: VectorStoreIndex for scenes

    Returns:
        Query response from the selected index
    """
    decision = route_query(query)
    print(decision)

    if decision.route == "book":
        query_engine = book_index.as_query_engine(similarity_top_k=1)

    elif decision.route == "chapter":
        if decision.chapter_index is not None:
            # apply metadata filtering
            query_engine = chapter_index_engine_filtered(chapter_index, decision.chapter_index)
        else:
            query_engine = chapter_index.as_query_engine()

    else:  # scene
        query_engine = scenes_index.as_query_engine(similarity_top_k=SCENE_SIMILARITY_TOP_K)

    return query_engine.query(query)


def display_sources(answer):
    sources = ""
    for i, node in enumerate(answer.source_nodes):
        md = node.metadata
        if "chapter_index" and "scene_index" in md.keys():
            sources += f"*Source {i+1}: chapter {md["chapter_index"]}, scene {md["scene_index"]}*  \n"
        elif "chapter_index" in md.keys():
            sources += f"*Source {i+1}: chapter {md["chapter_index"]}*  \n"
    return sources
