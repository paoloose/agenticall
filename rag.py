from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import getpass
import os

OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
  OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")

#os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class PromptAugmenter:
    def __init__(self, data_dir: str, db_path: str = "local_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self.load_or_create_vector_store()
        self.retriever = self.setup_retriever()

    def load_or_create_vector_store(self):
        if os.path.exists(self.db_path):
            print("Loading existing vector store...")
            return FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("Creating new vector store...")
            documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200,
            )
            texts = text_splitter.split_documents(documents)
            # SKLearnVectorStore
            # FAISS
            # ChromaDB
            vector_store = FAISS.from_documents(texts, self.embeddings)
            vector_store.save_local(self.db_path)
            return vector_store

    def load_documents(self) -> list[Document]:
        # <https://python.langchain.com/docs/how_to/document_loader_pdf/>
        print("Loading PDF files...")
        documents = []
        for file in os.listdir(self.data_dir):
            if not file.endswith(".pdf"):
                continue
            file_path = os.path.join(self.data_dir, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        return documents

    def setup_retriever(self):
        print("Setting up retriever...")
        return self.vector_store.as_retriever()

    def query(self, question: str) -> tuple[str, list[Document]]:
        print("Processing query...")
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)

        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create augmented prompt
        augmented_prompt = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

        return augmented_prompt, relevant_docs

    def display_relevant_docs(self, source_docs):
        print("\nRelevant documents:")
        for i, doc in enumerate(source_docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Content: {doc.page_content[:200]}...")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Page: {doc.metadata.get('page', 'Unknown')}")

def main():
    data_dir = "documents"

    rag = PromptAugmenter(data_dir)

    print("Ready to generate augmented prompts. Type 'quit' to exit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == "quit":
            break
        augmented_prompt, source_docs = rag.query(user_question)
        print("\nAugmented Prompt:")
        print(augmented_prompt)
        rag.display_relevant_docs(source_docs)
        print("\n" + "-"*50)

    print("thank you for using the PDF RAG system!")

if __name__ == "__main__":
    main()
