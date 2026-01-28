"""
Main chatbot script for Doctor Dolittle Q&A system.
"""
from dotenv import load_dotenv, find_dotenv
from src.indexing import build_all_indices
from src.retrieval import answer_query

# Load environment variables
load_dotenv(find_dotenv())


def main():
    """
    Main function to run the Doctor Dolittle chatbot.
    """
    book_index, chapter_index, scenes_index = build_all_indices()

    print("\nDoctor Dolittle Chatbot - Ready to answer your questions!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_query = input("Your question: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_query:
            continue

        try:
            response = answer_query(user_query, book_index, chapter_index, scenes_index)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()