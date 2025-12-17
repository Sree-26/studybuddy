import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Fixed in previous step
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# UPDATED IMPORTS HERE
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db"

def load_documents(files):
    documents = []
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # Save uploaded file to specific temp directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            elif file_extension == ".pptx":
                loader = UnstructuredPowerPointLoader(temp_file_path)
                documents.extend(loader.load())
            else:
                # Fallback for unsupported types, though UI should limit this
                print(f"Unsupported file type: {file_extension}")
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp dir {temp_dir}: {e}")
            
    return documents

def clean_text(text):
    # Basic cleaning - removing excessive whitespace
    if text:
        return " ".join(text.split())
    return ""

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Pre-clean text in documents
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Creates a new vector store. 
    CRITICAL FIX: Deletes the existing DB to prevent duplicate data.
    """
    
    # 1. Check if the DB exists and delete it
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            print(f"Old vector store at {PERSIST_DIRECTORY} deleted. Starting fresh.")
        except OSError as e:
            print(f"Error: Could not delete old vector store. {e}")

    # 2. Re-initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 3. Create the new store (This will create the directory again)
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    return vectorstore

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")

    system_prompt = (
        "You are an assistant for curious engineering students. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain





