from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import getpass
import os

OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
  OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")

#os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RAGPromptAugmenter:
    def __init__(self, data_dir: str, db_path: str = "local_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
        )
        self.rag_validator = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            api_key=OPENAI_API_KEY,
        )
        self.vector_store = self.load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever()

    def load_or_create_vector_store(self):
        # <https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html>
        # <https://github.com/facebookresearch/faiss>
        print("Initializing vector store and generating document embeddings...")
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=200,
        )
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, self.embeddings)
        vector_store.save_local(self.db_path)
        return vector_store

    def load_documents(self) -> list[Document]:
        # <https://python.langchain.com/docs/how_to/document_loader_pdf/>
        print("Loading PDF files for embedding generation...")
        documents = []
        for file in os.listdir(self.data_dir):
            if not file.endswith(".pdf"):
                continue
            file_path = os.path.join(self.data_dir, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        return documents

    def validate_relevance(self, question: str, retrieved_docs: list[Document]) -> bool:
        """
        Validate if the retrieved documents are relevant to the user's question.
        Returns True if relevant, False otherwise.
        """
        # Create a summary of the retrieved documents
        context_summary = "\n".join([doc.page_content for doc in retrieved_docs])

        validation_prompt = f"""You are a relevance validator. Your task is to determine if the provided context is relevant to answer the user's question.

<question>
{question}
</question>

<context>
{context_summary}
</context>

<instruction>
- Answer ONLY with "YES" if the context contains information that can help answer the question
- Answer ONLY with "NO" if the context does not contain relevant information to answer the question
- Be strict in your evaluation - if the context is only tangentially related, answer "NO"
</instructions>

Response:"""

        try:
            response = self.rag_validator.invoke(
                input=[
                    SystemMessage("You are a relevance validator that responds only with YES or NO."),
                    HumanMessage(validation_prompt),
                ],
            )
            answer = response.content
            return answer.upper().strip() == "YES"

        except Exception as e:
            print(f"Error in relevance validation: {e}")
            # If validation fails, default to returning the context
            return True

    def query(self, question: str) -> tuple[str, list[Document]]:
        print("Processing query with semantic similarity search...")
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)

        # Validate if the retrieved documents are relevant
        is_relevant = self.validate_relevance(question, relevant_docs)

        if not is_relevant:
            print("Retrieved documents are not relevant to the question.")
            return "No hay información disponible para la consulta", relevant_docs

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

    rag = RAGPromptAugmenter(data_dir)

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
