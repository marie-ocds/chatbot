"""
Data loading module for processing the PDF book.
"""
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from src.config import DATA_PATH, BOOK_TITLE, SCENE_CHUNK_SIZE, SCENE_CHUNK_OVERLAP, LLM_MODEL
from src.utils import extract_pdf_text, clean_text, extract_chapters


def load_book_data():
    """
    Load and process the PDF book data.

    Returns:
        tuple: (full_text_clean, chapters, text_chapters)
            - full_text_clean: Cleaned text of the entire book
            - chapters: Dictionary mapping chapter index to chapter title
            - text_chapters: List of chapter texts split by double newlines
    """
    # Extract text from PDF
    full_text = extract_pdf_text(DATA_PATH)

    # Clean the text
    full_text_clean = clean_text(full_text)

    # Extract chapter information
    chapters = extract_chapters(full_text_clean)

    # Split into chapters (skip first element which is likely front matter)
    text_chapters = full_text_clean.split("\n\n")[1:]

    return full_text_clean, chapters, text_chapters


def create_book_document(full_text_clean):
    """
    Create a Document object for the entire book.

    Args:
        full_text_clean: Cleaned text of the entire book

    Returns:
        Document: LlamaIndex Document with book metadata
    """
    return Document(
        text=full_text_clean,
        metadata={"book_title": BOOK_TITLE}
    )


def create_chapter_documents(text_chapters, chapters):
    """
    Create Document objects for each chapter.

    Args:
        text_chapters: List of chapter texts
        chapters: Dictionary mapping chapter index to chapter title

    Returns:
        list: List of Document objects with chapter metadata
    """
    chapter_documents = []

    for chapter_idx, chapter_text in enumerate(text_chapters):
        chapter_documents.append(
            Document(
                text=chapter_text,
                metadata={
                    "book_title": BOOK_TITLE,
                    "chapter_index": chapter_idx + 1,
                    "chapter_title": chapters.get(chapter_idx + 1, "")
                }
            )
        )

    return chapter_documents


def create_scene_documents(chapter_documents):
    """
    Create scene-level documents by splitting chapters into smaller chunks.

    Args:
        chapter_documents: List of Document objects for chapters

    Returns:
        list: List of scene-level document nodes
    """
    scene_splitter = SentenceSplitter(
        chunk_size=SCENE_CHUNK_SIZE,
        chunk_overlap=SCENE_CHUNK_OVERLAP
    )

    scenes_documents = []

    for chapter_index, chapter_doc in enumerate(chapter_documents):
        nodes = scene_splitter.get_nodes_from_documents([chapter_doc])

        for i, node in enumerate(nodes):
            node.metadata.update({
                "book_title": BOOK_TITLE,
                "chapter_index": chapter_doc.metadata["chapter_index"],
                "chapter_title": chapter_doc.metadata["chapter_title"],
                "scene_index": i
            })
            scenes_documents.append(node)

    return scenes_documents


def create_book_summary_document(text_chapters, chapters):
    """
       Create an overall summary of the book by concatenating all the chapter summaries.

       Args:
           text_chapters: List of chapter texts
           chapters: Dictionary mapping chapter index to chapter title

       Returns:
           Text: Overall summary of the book
       """
    llm = OpenAI(model=LLM_MODEL)
    chapter_summaries = []
    print("Generating global summary...")

    for chapter_idx, chapter_text in enumerate(text_chapters):
        chapter_title = chapters.get(chapter_idx + 1, "")
        summary = llm.complete(f"Summarize this part of the plot in a few sentences: {chapter_text}")
        chapter_summaries.append(f"Summary of chapter {chapter_idx+1}: {chapter_title} \n {summary.text}")

    global_summary = "The Story of Doctor Dolittle, By Hugh Lofting\n\n" + "\n\n".join(chapter_summaries)
    print("Book length reduced to {} words!".format(len(global_summary.split())))

    return Document(
        text=global_summary,
        metadata={"book_title": BOOK_TITLE}
    )