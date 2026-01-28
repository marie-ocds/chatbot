"""
Indexing module for creating and managing vector store indices.
"""
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from src.config import (
    CHROMA_PATH,
    LLM_MODEL,
    COLLECTION_BOOK,
    COLLECTION_CHAPTERS,
    COLLECTION_SCENES)


def initialize_settings():
    """
    Initialize global LLM and embedding model settings.
    """
    Settings.embed_model = OpenAIEmbedding()
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0)


def get_chroma_client():
    """
    Get or create ChromaDB persistent client.

    Returns:
        chromadb.PersistentClient: ChromaDB client instance
    """
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_or_create_index(db, collection_name, documents=None):
    """
    Generic function to create or load a vector store index.

    Args:
        db: ChromaDB client
        collection_name: Name of the ChromaDB collection
        documents: Documents to index (required only if creating new index)

    Returns:
        VectorStoreIndex: The created or loaded index
    """
    collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if collection already has data
    if collection.count() > 0:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        # Create new index
        if documents is None:
            raise ValueError(f"Documents required to create new index for {collection_name}")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )

    return index


def build_all_indices():
    """
    Build or load all vector store indices (book, chapters, scenes).

    If indices already exist in ChromaDB, they will be loaded.
    Otherwise, new indices will be created.

    Returns:
        tuple: (book_index, chapter_index, scenes_index)
    """
    from src.data_loader import (
        load_book_data,
        create_book_summary_document,
        create_chapter_documents,
        create_scene_documents
    )

    # Initialize settings
    initialize_settings()

    # Get ChromaDB client
    db = get_chroma_client()

    # Check if we need to load data (only if creating new indices)
    book_collection = db.get_or_create_collection(name=COLLECTION_BOOK)
    chapter_collection = db.get_or_create_collection(name=COLLECTION_CHAPTERS)
    scenes_collection = db.get_or_create_collection(name=COLLECTION_SCENES)

    # Check if indices exist
    indices_exist = (book_collection.count() > 0 and
                     chapter_collection.count() > 0 and
                     scenes_collection.count() > 0)

    # Load or create documents
    if indices_exist:
        print("Indices loaded successfully")
        # we don't need to open and preprocess the data
        book_document = None
        chapter_documents = None
        scenes_documents = None

    else:
        print("Building indices for the first time...")
        full_text_clean, chapters, text_chapters = load_book_data()
        book_document = create_book_summary_document(text_chapters, chapters)
        chapter_documents = create_chapter_documents(text_chapters, chapters)
        scenes_documents = create_scene_documents(chapter_documents)

    # Create or load indices using the generic function
    book_index = get_or_create_index(
        db, COLLECTION_BOOK,
        [book_document] if book_document else None
    )
    chapter_index = get_or_create_index(
        db, COLLECTION_CHAPTERS,
        chapter_documents
    )
    scenes_index = get_or_create_index(
        db, COLLECTION_SCENES,
        scenes_documents
    )

    return book_index, chapter_index, scenes_index