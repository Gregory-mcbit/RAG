from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil



class Dataloader:
    _CHROMA_PATH = "chroma"
    _DATA_PATH = "./Data"


    def __init__(self):
        load_dotenv()
        openai.api_key = os.environ['OPENAI_API_KEY']
    

    def load_data(self):
        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._save_to_chroma(chunks)
    

    def _load_documents(self):
        loader = DirectoryLoader(self._DATA_PATH, glob="*.md")
        documents = loader.load()

        return documents


    def _split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

        return chunks


    def _save_to_chroma(self, chunks: list[Document]):
        # Clear out the database first.
        if os.path.exists(self._CHROMA_PATH):
            shutil.rmtree(self._CHROMA_PATH)

        # Create a new DB from the documents.
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self._CHROMA_PATH
        )

        db.persist()
        print(f"Saved {len(chunks)} chunks to {self._CHROMA_PATH}.")


if __name__ == "__main__":
    dataloader = Dataloader()
    dataloader.load_data()
