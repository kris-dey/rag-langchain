from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

# New variable names
CHROMA_DIRECTORY = "chroma_db"
DOCUMENTS_DIRECTORY = "text_documents"

def main():
    # Load documents from a directory.
    documents = load_text_documents()
    
    text_chunks = split_documents_into_chunks(documents)
    
    # Create a new Chroma database from the chunks.
    chroma_db = Chroma.from_documents(
        text_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DIRECTORY,
    )

def load_text_documents():
    document_loader = DirectoryLoader(DOCUMENTS_DIRECTORY, glob="*.md")
    
    # Load documents from the specified directory.
    documents = document_loader.load()
    
    return documents

def split_documents_into_chunks(documents: list[Document]):
    # Initialize text splitter with the desired parameters.
    text_splitter = RecursiveCharacterTextSplitter(
        add_start_index=True,
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)
    


if __name__ == "__main__":
    main()