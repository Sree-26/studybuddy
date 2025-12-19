import os
import shutil
import tempfile
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 3
MIN_CHUNK_LENGTH = 50  # Minimum characters for a valid chunk

class RAGEngineError(Exception):
    """Custom exception for RAG engine errors"""
    pass

def load_documents(files):
    """
    Load documents from uploaded files with improved error handling.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        List of loaded documents
        
    Raises:
        RAGEngineError: If no documents could be loaded
    """
    documents = []
    errors = []
    temp_dirs = []
    
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        temp_file_path = os.path.join(temp_dir, file.name)
        
        try:
            # Save uploaded file
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Load based on file type
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded PDF: {file.name} ({len(docs)} pages)")
                
            elif file_extension == ".pptx":
                loader = UnstructuredPowerPointLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded PowerPoint: {file.name} ({len(docs)} slides)")
                
            else:
                error_msg = f"Unsupported file type: {file_extension} for file {file.name}"
                logger.warning(error_msg)
                errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Error loading {file.name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Clean up temp directories after all loading is complete
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp dir {temp_dir}: {e}")
    
    # Validate we got some documents
    if not documents:
        error_summary = "\n".join(errors) if errors else "No valid documents found"
        raise RAGEngineError(f"Failed to load any documents. Errors:\n{error_summary}")
    
    if errors:
        logger.warning(f"Loaded {len(documents)} documents with {len(errors)} errors")
    
    return documents

def clean_text(text: str) -> str:
    """
    Enhanced text cleaning to remove common PDF/PowerPoint artifacts.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove standalone page numbers (simple heuristic)
    text = re.sub(r'\b\d{1,3}\b(?=\s|$)', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '--', text)
    
    # Remove URLs (optional - comment out if you want to keep them)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Strip and normalize
    text = text.strip()
    
    return text

def is_valid_chunk(chunk_text: str, min_length: int = MIN_CHUNK_LENGTH) -> bool:
    """
    Validate if a chunk contains meaningful content.
    
    Args:
        chunk_text: Text chunk to validate
        min_length: Minimum character length
        
    Returns:
        True if chunk is valid, False otherwise
    """
    if not chunk_text or len(chunk_text) < min_length:
        return False
    
    # Check if chunk has a reasonable ratio of alphanumeric characters
    alphanumeric_count = sum(c.isalnum() for c in chunk_text)
    if alphanumeric_count / len(chunk_text) < 0.3:  # At least 30% meaningful characters
        return False
    
    return True

def chunk_documents(documents, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Split documents into chunks with validation.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of valid document chunks
        
    Raises:
        RAGEngineError: If no valid chunks are created
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Clean text in documents
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    # Filter out invalid chunks
    valid_chunks = [chunk for chunk in chunks if is_valid_chunk(chunk.page_content)]
    
    invalid_count = len(chunks) - len(valid_chunks)
    if invalid_count > 0:
        logger.info(f"Filtered out {invalid_count} invalid chunks")
    
    if not valid_chunks:
        raise RAGEngineError("No valid chunks created after filtering")
    
    logger.info(f"Created {len(valid_chunks)} valid chunks")
    return valid_chunks

def create_vector_store(chunks):
    """
    Creates a new vector store with proper cleanup.
    
    Args:
        chunks: Document chunks to embed
        
    Returns:
        Chroma vectorstore instance
        
    Raises:
        RAGEngineError: If vector store creation fails
    """
    try:
        # Delete existing database
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                logger.info(f"Deleted old vector store at {PERSIST_DIRECTORY}")
            except OSError as e:
                raise RAGEngineError(f"Could not delete old vector store: {e}")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create new vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        
        logger.info(f"Vector store created with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        raise RAGEngineError(f"Failed to create vector store: {str(e)}")

def get_rag_chain(vectorstore, retrieval_k: int = DEFAULT_RETRIEVAL_K, 
                  system_prompt: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile"):
    """
    Create RAG chain with configurable parameters.
    
    Args:
        vectorstore: Chroma vectorstore instance
        retrieval_k: Number of documents to retrieve
        system_prompt: Custom system prompt (optional)
        model_name: LLM model name
        
    Returns:
        RAG chain instance
        
    Raises:
        RAGEngineError: If chain creation fails or API key is missing
    """
    try:
        # Check for API key
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RAGEngineError(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it using: export GROQ_API_KEY='your-key-here'"
            )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        )
        
        # Initialize LLM with error handling
        llm = ChatGroq(
            model_name=model_name,
            groq_api_key=groq_api_key,
            temperature=0.1,
            max_retries=3
        )
        
        # Use custom or default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an assistant for curious engineering students. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logger.info("RAG chain created successfully")
        return rag_chain
        
    except Exception as e:
        raise RAGEngineError(f"Failed to create RAG chain: {str(e)}")

def query_rag(rag_chain, question: str) -> dict:
    """
    Query the RAG system with error handling.
    
    Args:
        rag_chain: RAG chain instance
        question: User question
        
    Returns:
        Dictionary with answer and source documents
        
    Raises:
        RAGEngineError: If query fails
    """
    try:
        result = rag_chain.invoke({"input": question})
        logger.info(f"Query processed: {question[:50]}...")
        return result
    except Exception as e:
        raise RAGEngineError(f"Query failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    """
    Example usage of the RAG engine.
    Set GROQ_API_KEY environment variable before running.
    """
    try:
        # This is a placeholder - in actual use, you'd get files from Streamlit
        # files = st.file_uploader("Upload documents", type=["pdf", "pptx"], accept_multiple_files=True)
        
        # Load and process documents
        # documents = load_documents(files)
        # chunks = chunk_documents(documents)
        # vectorstore = create_vector_store(chunks)
        # rag_chain = get_rag_chain(vectorstore)
        
        # Query the system
        # response = query_rag(rag_chain, "What is the main topic?")
        # print(response["answer"])
        
        print("RAG Engine ready. Import and use the functions in your Streamlit app.")
        
    except RAGEngineError as e:
        logger.error(f"RAG Engine Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")






