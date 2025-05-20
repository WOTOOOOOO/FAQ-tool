import os

from pypdf import PdfReader

from langchain.schema import Document

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class RegulationService:
    @staticmethod
    def setup(pdf_path: str, index_path: str, number_of_chunks: int = 25):
        regulation_index_result = RegulationService.create_regulations_vector_index(
            pdf_path,
            index_path,
            number_of_chunks
        )
        return regulation_index_result

    @staticmethod
    def create_regulations_vector_index(pdf_path: str, index_path: str, number_of_chunks: int = 25):
        """
        Processes a PDF file, splits its content into semantic chunks, and creates a FAISS vector index.

        Args:
            pdf_path (str): Path to the regulations PDF.
            index_path (str): Directory path to save the FAISS index.
            number_of_chunks (int): Number of semantic chunks to create.
        Returns:
            str: A message indicating that the FAISS index was created or not
        """

        if os.path.exists(index_path):
            return "Index already exists"

        # Extract text from the PDF
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Create semantic chunks
        embeddings = HuggingFaceEmbeddings()
        text_splitter = SemanticChunker(embeddings, number_of_chunks=number_of_chunks)
        docs = text_splitter.create_documents([text])

        # Wrap into LangChain Documents
        documents = [Document(page_content=doc.page_content) for doc in docs]

        # Embed and save FAISS index
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)

        return "Index created"
