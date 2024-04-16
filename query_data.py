import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# New variable names
CHROMA_DIRECTORY = "chroma_db"
DEFAULT_PROMPT_TEMPLATE = """
Here is the context:

{context}

---

Answer this based on the context: {question}
"""


def main():
    query_text = parse_command_line_arguments()

    # Initialize and prepare the Chroma database
    db = initialize_chroma_database()

    # Search the database and retrieve results
    results = search_database(db, query_text)
    
    # If no relevant results found, exit grsacefully
    if not results or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    context_text = prepare_context_text(results)
    
    # Create prompt using context and query text
    prompt = create_prompt(context_text, query_text)
    print(prompt)

    # Generate response using ChatOpenAI model
    response_text = generate_response(prompt)

    # Format and display response along with sources
    display_response(results, response_text)


def parse_command_line_arguments():
    # Create a command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text


def initialize_chroma_database():
    # Initialize the Chroma database with OpenAI embeddings
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embedding_function)
    return db


def search_database(db, query_text):
    # Perform a similarity search on the database with relevance scores
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    return results


def prepare_context_text(results):
    # Extract context text from search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text


def create_prompt(context_text, query_text):
    # Format the prompt template with context and query text
    prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt


def generate_response(prompt):
    # Generate response using ChatOpenAI model
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text


def display_response(results, response_text):
    # Extract sources from search results
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    # Format and display response along with sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()