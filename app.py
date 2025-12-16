import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import time
from dotenv import load_dotenv
from rag_engine import load_documents, chunk_documents, create_vector_store, get_rag_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Study Buddy RAG", layout="wide")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

init_session_state()

st.title("📚 Study Buddy")

with st.sidebar:
    
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or PPTX files", 
        type=["pdf", "pptx"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one file.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("Please ensure Groq API Key is set.")
        else:
            with st.status("Processing documents...", expanded=True) as status:
                st.write("Loading documents...")
                raw_docs = load_documents(uploaded_files)
                
                st.write("Chunking text...")
                chunks = chunk_documents(raw_docs)
                
                st.write("Creating vector store...")
                vector_store = create_vector_store(chunks)
                st.session_state.vector_store = vector_store
                
                st.write("Initializing RAG chain...")
                st.session_state.rag_chain = get_rag_chain(vector_store)
                
                status.update(label="System Ready!", state="complete", expanded=False)
            st.success("Documents processed successfully!")

# Main Chat Interface
if st.session_state.rag_chain is None:
    st.info("👈 Please upload documents and click 'Process Documents' to start.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        source_name = doc.metadata.get("source", "Unknown")
                        page_num = doc.metadata.get("page", "Unknown")
                        # Basic display
                        st.markdown(f"**Source {i+1}:** {os.path.basename(source_name)} (Page {page_num})")
                        st.text(doc.page_content[:200] + "...")

    # Chat input
    if prompt := st.chat_input("Ask a question based on your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = st.session_state.rag_chain.invoke({"input": prompt})
                elapsed_time = time.time() - start_time
                
                answer = response["answer"]
                sources = response["context"]
                
                st.markdown(answer)
                st.caption(f"Latency: {elapsed_time:.2f}s")
                
                with st.expander("View Sources"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get("source", "Unknown")
                        # Handle different metadata keys if needed, PyPDF usually has 'source' and 'page'
                        page_num = doc.metadata.get("page", "ND")  
                        st.markdown(f"**Source {i+1}:** {os.path.basename(source_name)} (Page {page_num})")
                        st.text(doc.page_content[:300] + "...")
                        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })


