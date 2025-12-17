import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Deployment-Ready Constants
# We use a temp directory so this works on Read-Only cloud servers
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = os.path.join(tempfile.gettempdir(), "chroma_db_study_buddy")

# --- BACKEND FUNCTIONS (Your Logic) ---

def load_documents(files):
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for file in files:
            file_extension = os.path.splitext(file.name)[1].lower()
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
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
                
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # Simple cleaning inside the chunking
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())
        
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    # Clear existing DB to prevent duplicates/locking
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except Exception as e:
            st.warning(f"Note: Could not clear old DB ({e}). Continuing...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Ensure you have GROQ_API_KEY in your .env or secrets
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    system_prompt = (
        "You are 'Study Buddy', an intelligent tutor for students. "
        "Answer the question strictly based on the provided context. "
        "If the answer is not in the context, say 'I couldn't find that in your notes'. "
        "Keep answers concise and helpful."
        "\n\n"
        "Context: {context}"
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

# --- FRONTEND (Streamlit UI) ---

def main():
    st.set_page_config(page_title="AI Study Buddy", page_icon="📚")
    
    st.title("📚 AI Study Buddy")
    st.caption("Upload your PDFs/PPTs and ask questions to prepare for exams.")

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar for Uploads
    with st.sidebar:
        st.header("📂 Your Notes")
        uploaded_files = st.file_uploader(
            "Upload Lecture Materials", 
            type=["pdf", "pptx"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Notes"):
            if uploaded_files:
                with st.spinner("Analyzing your documents..."):
                    # 1. Load
                    raw_docs = load_documents(uploaded_files)
                    st.info(f"Loaded {len(raw_docs)} pages.")
                    
                    # 2. Chunk
                    chunks = chunk_documents(raw_docs)
                    st.info(f"Split into {len(chunks)} chunks.")
                    
                    # 3. Embed & Store
                    vectorstore = create_vector_store(chunks)
                    
                    # Store in session state so we don't reload on every interaction
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Knowledge Base Ready!")
            else:
                st.warning("Please upload files first.")

    # Chat Interface
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your notes..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        if st.session_state.vectorstore is not None:
            [Image of RAG retrieval process]

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    rag_chain = get_rag_chain(st.session_state.vectorstore)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("⚠️ Please upload and process documents first!")

if __name__ == "__main__":
    main()





