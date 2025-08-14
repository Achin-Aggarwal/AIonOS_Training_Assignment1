import streamlit as st
import os
import tempfile
import shutil
from typing import List, Optional
import hashlib
import time
from pathlib import Path

# Updated imports to match your RAG code
from resume_rag_ollama import (
    setup_dynamic_system, 
    run_qna_pipeline, 
    qna_system_message, 
    qna_user_message_template,
    cleanup_temp_files,
    create_pdf_hash,
    RAGConfig,
    ModelProvider,
    create_openai_config,
    create_ollama_config,
    validate_environment,
    OLLAMA_MODEL_NAME,
    OPENAI_MODEL_NAME
)

st.set_page_config(
    page_title="Resume RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.95)), 
                    url('https://images.unsplash.com/photo-1497366216548-37526070297c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2069&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .status-success {
        padding: 1.2rem;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.15);
        color: #065f46;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-info {
        padding: 1.2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        color: #1e3a8a;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .status-warning {
        padding: 1.2rem;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
        color: #92400e;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .response-container {
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.85) 100%);
        border-radius: 16px;
        border: 1px solid rgba(203, 213, 225, 0.5);
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        color: #1e293b;
        backdrop-filter: blur(10px);
    }
    
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 250, 252, 0.7) 100%);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid rgba(203, 213, 225, 0.3);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        color: #334155;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.8) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(203, 213, 225, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        color: #475569;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(203, 213, 225, 0.4);
        border-radius: 12px;
        color: #1e293b;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(203, 213, 225, 0.4);
        border-radius: 8px;
        color: #1e293b;
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8);
        border: 2px dashed rgba(148, 163, 184, 0.5);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .sidebar .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        color: #1e293b;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .stMarkdown {
        color: #334155;
    }
    
    .sidebar .stMarkdown {
        color: #475569;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        color: #1e3a8a;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        color: #065f46;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        color: #92400e;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'uploaded_files_hash' not in st.session_state:
        st.session_state.uploaded_files_hash = None
    if 'processing_mode' not in st.session_state:
        st.session_state.processing_mode = "uploaded"
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    if 'filenames' not in st.session_state:
        st.session_state.filenames = []
    if 'config' not in st.session_state:
        # Initialize with default config based on environment
        env_status = validate_environment()
        if env_status["openai_available"]:
            st.session_state.config = create_openai_config()
        else:
            st.session_state.config = create_ollama_config()

def cleanup_session():
    """Clean up session state and temporary files"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.success("🗑️ Cleaned up temporary files")
        except Exception as e:
            st.warning(f"⚠️ Could not clean up temp files: {e}")
    
    # Clean up temporary database directories
    temp_db_dir = f"./temp_dbs/session_{st.session_state.session_id}"
    if os.path.exists(temp_db_dir):
        try:
            shutil.rmtree(temp_db_dir)
        except Exception as e:
            st.warning(f"⚠️ Could not clean up temp database: {e}")
    
    # Reset session state
    st.session_state.system_ready = False
    st.session_state.llm = None
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    st.session_state.temp_dir = None
    st.session_state.uploaded_files_hash = None
    st.session_state.filenames = []

def display_status_message(message: str, status_type: str = "info"):
    """Display styled status messages"""
    css_class = f"status-{status_type}"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def validate_pdf_files(uploaded_files: List) -> tuple[bool, str]:
    """Validate uploaded PDF files"""
    if not uploaded_files:
        return False, "No files uploaded"
    
    for file in uploaded_files:
        if not file.name.lower().endswith('.pdf'):
            return False, f"File '{file.name}' is not a PDF"
        
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            return False, f"File '{file.name}' is too large (max 50MB)"
    
    return True, "Valid PDF files"

def format_file_info(uploaded_files: List) -> str:
    """Format file information for display"""
    if not uploaded_files:
        return "No files"
    
    total_size = sum(file.size for file in uploaded_files)
    size_mb = total_size / (1024 * 1024)
    
    file_list = ", ".join([file.name for file in uploaded_files])
    return f"{len(uploaded_files)} files ({size_mb:.1f}MB): {file_list}"

def setup_system_with_files(uploaded_files: List = None, use_default: bool = False):
    """Setup the RAG system with files or default folder"""
    with st.spinner("🔄 Setting up RAG system..."):
        try:
            if use_default:
                st.session_state.processing_mode = "default"
                result = setup_dynamic_system(config=st.session_state.config)
            else:
                st.session_state.processing_mode = "uploaded"
                result = setup_dynamic_system(
                    uploaded_files=uploaded_files,
                    session_id=st.session_state.session_id,
                    config=st.session_state.config
                )
            
            # Handle the result tuple properly
            if result and len(result) == 5:
                llm, retriever, vectorstore, temp_dir, filenames = result
            else:
                return False, "Invalid system setup result"
            
            if llm and retriever and vectorstore:
                st.session_state.llm = llm
                st.session_state.retriever = retriever
                st.session_state.vectorstore = vectorstore
                st.session_state.temp_dir = temp_dir
                st.session_state.system_ready = True
                st.session_state.filenames = filenames or []
                
                if uploaded_files:
                    st.session_state.uploaded_files_hash = create_pdf_hash(uploaded_files)
                
                return True, "System setup successful!"
            else:
                return False, "Failed to setup system components. Check configuration and dependencies."
                
        except Exception as e:
            return False, f"Error setting up system: {str(e)}"

def handle_user_query(user_question: str) -> str:
    """Handle user query and return response"""
    if not st.session_state.system_ready:
        return "❌ System not ready. Please setup the system first."
    
    try:
        with st.spinner("🧠 Analyzing documents and generating response..."):
            response = run_qna_pipeline(
                user_input=user_question,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm,
                config=st.session_state.config
            )
            return response
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def display_predefined_queries():
    """Display predefined query buttons"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Complete Summary", use_container_width=True):
            return "Provide a comprehensive summary of all documents"
    
    with col2:
        if st.button("🧠 Generate Quiz", use_container_width=True):
            return "Generate comprehensive quiz questions about the content"
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def display_model_configuration():
    """Display model configuration options in sidebar"""
    st.markdown("### 🤖 Model Configuration")
    
    env_status = validate_environment()
    
    # Model provider selection
    available_providers = []
    if env_status["openai_available"]:
        available_providers.append("OpenAI")
    available_providers.append("Ollama")
    
    if len(available_providers) > 1:
        provider_choice = st.selectbox(
            "Model Provider:",
            available_providers,
            index=0 if env_status["openai_available"] else available_providers.index("Ollama")
        )
    else:
        provider_choice = available_providers[0]
        st.info(f"Using {provider_choice} (only available option)")
    
    # Update config based on selection
    if provider_choice == "OpenAI" and env_status["openai_available"]:
        model_name = st.selectbox(
            "OpenAI Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        st.session_state.config = create_openai_config(model=model_name)
    else:
        model_name = st.selectbox(
            "Ollama Model:",
            ["gemma3:1b", "llama3:latest", "mistral:latest", "phi3:latest"],
            index=0
        )
        st.session_state.config = create_ollama_config(model=model_name)
    
    # Display current configuration
    st.markdown("**Current Config:**")
    st.code(f"""
Provider: {st.session_state.config.model_provider}
Model: {model_name}
Embedding: {st.session_state.config.embedding_provider}
Temperature: {st.session_state.config.temperature}
    """)
    
    return env_status

def main():
    """Main application function"""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">📄 Advanced Resume Analysis System</h1>', unsafe_allow_html=True)
    
    # Display session info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_model = (st.session_state.config.openai_model 
                        if st.session_state.config.model_provider == ModelProvider.OPENAI 
                        else st.session_state.config.ollama_model)
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <strong>🤖 Model:</strong> {current_model}<br>
            <strong>🔗 Session:</strong> {st.session_state.session_id}<br>
            <strong>📊 Mode:</strong> {st.session_state.processing_mode.title()}
        </div>
        """, unsafe_allow_html=True)
    
    st.info("💡 **Enhanced System:** Advanced document analysis with comprehensive summaries, intelligent Q&A, and detailed assessments. No chat history saved - each query processed independently.")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
        
        # Model configuration
        env_status = display_model_configuration()
        
        st.markdown("---")
        
        # Setup mode selection
        setup_mode = st.radio(
            "Choose setup mode:",
            ["📤 Upload PDFs", "📁 Use Default Folder"],
            index=0
        )
        
        st.markdown("---")
        
        if setup_mode == "📤 Upload PDFs":
            st.markdown("### 📤 Upload PDF Files")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files (max 50MB each)"
            )
            
            if uploaded_files:
                is_valid, message = validate_pdf_files(uploaded_files)
                
                if is_valid:
                    st.success(f"✅ {format_file_info(uploaded_files)}")
                    
                    current_hash = create_pdf_hash(uploaded_files)
                    files_changed = current_hash != st.session_state.uploaded_files_hash
                    
                    if files_changed or not st.session_state.system_ready:
                        if st.button("🚀 Setup System", use_container_width=True, type="primary"):
                            if st.session_state.system_ready:
                                cleanup_session()
                            
                            success, setup_message = setup_system_with_files(uploaded_files)
                            
                            if success:
                                st.success(setup_message)
                                st.rerun()
                            else:
                                st.error(setup_message)
                    else:
                        st.info("✅ System ready with current files")
                        if st.session_state.filenames:
                            st.markdown("**Processed Files:**")
                            for filename in st.session_state.filenames:
                                st.markdown(f"• {filename}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.info("Please upload PDF files to continue")
        
        else:  # Default folder mode
            st.markdown("### 📁 Default Folder Mode")
            st.info("Using PDFs from the 'Resume' folder or existing database")
            
            if not st.session_state.system_ready or st.session_state.processing_mode != "default":
                if st.button("🚀 Setup System", use_container_width=True, type="primary"):
                    if st.session_state.system_ready:
                        cleanup_session()
                    
                    success, setup_message = setup_system_with_files(use_default=True)
                    
                    if success:
                        st.success(setup_message)
                        st.rerun()
                    else:
                        st.error(setup_message)
            else:
                st.success("✅ System ready with default folder")
        
        st.markdown("---")
        
        # System status display
        if st.session_state.system_ready:
            st.success("🟢 System Active")
            st.info(f"Mode: {st.session_state.processing_mode.title()}")
            if st.session_state.filenames:
                with st.expander("📄 Loaded Documents"):
                    for filename in st.session_state.filenames:
                        st.markdown(f"• **{filename}**")
        else:
            st.warning("🟡 System Inactive")
        
        # Environment status
        with st.expander("🔧 Environment Status"):
            if env_status["openai_available"]:
                st.success("✅ OpenAI API Available")
            else:
                st.warning("⚠️ OpenAI API Key Not Set")
            
            if env_status["langchain_available"]:
                st.success("✅ LangChain Tracing Available")
                if env_status["langchain_project"]:
                    st.info(f"Project: {env_status['langchain_project']}")
            else:
                st.info("ℹ️ LangChain Tracing Not Configured")
        
        # Reset system button
        if st.session_state.system_ready:
            st.markdown("---")
            if st.button("🔄 Reset System", use_container_width=True):
                cleanup_session()
                st.success("System reset successfully!")
                st.rerun()
    
    # Main content area
    if st.session_state.system_ready:
        # Quick actions
        predefined_query = display_predefined_queries()
        
        if predefined_query:
            st.markdown(f"### 🤔 Query: {predefined_query}")
            response = handle_user_query(predefined_query)
            
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI Analysis")
            st.markdown(response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Custom query form
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💭 Custom Analysis")
        
        with st.form("query_form", clear_on_submit=True):
            user_question = st.text_area(
                "Enter your detailed question or analysis request:",
                placeholder="e.g., Compare the leadership experience across all candidates and provide detailed recommendations...",
                height=120
            )
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col3:
                submitted = st.form_submit_button("🚀 Analyze", use_container_width=True, type="primary")
            
            if submitted and user_question.strip():
                st.markdown(f"### 🤔 Analysis Request: {user_question.strip()}")
                response = handle_user_query(user_question.strip())
                
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("### 🤖 Comprehensive Analysis")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif submitted:
                st.warning("Please enter a question!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Getting started section
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 📤 Upload Mode
            1. Select "Upload PDFs" in the sidebar
            2. Configure your preferred model (OpenAI/Ollama)
            3. Upload your PDF files (resumes, documents, etc.)
            4. Click "Setup System" to process the files
            5. Start comprehensive analysis!
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 📁 Default Folder Mode  
            1. Place PDF files in the 'Resume' folder
            2. Select "Use Default Folder" in the sidebar
            3. Configure your preferred model (OpenAI/Ollama)
            4. Click "Setup System" to load existing database
            5. Start comprehensive analysis!
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example queries and system info
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Example Analysis Requests")
        st.markdown("""
        - **"Provide comprehensive summaries of all documents"**
        - **"What programming languages and technical skills are mentioned?"**
        - **"Compare work experience and career progression across candidates"**
        - **"Analyze educational backgrounds and certifications"**
        - **"What are the key strengths and achievements of each candidate?"**
        - **"Generate detailed quiz questions covering all important areas"**
        - **"Which candidates have leadership or management experience?"**
        - **"Compare salary expectations or availability information"**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System requirements
        with st.expander("⚙️ System Requirements & Features"):
            st.markdown("""
            **Prerequisites:**
            - **For Ollama**: Ollama installed and running (`ollama serve`)
            - **Models downloaded**: `ollama pull gemma3:1b` (or your preferred model)
            - **For OpenAI**: Valid API key in environment variables
            - **Required packages**: All dependencies from resume_rag_ollama.py
            
            **Enhanced Features:**
            - Advanced document analysis with structured summaries
            - Multi-format quiz generation (12-15 questions minimum)
            - Professional candidate assessment capabilities
            - Comparative analysis and ranking systems
            - Original filename preservation and display
            - Multi-model support (OpenAI GPT models and Ollama models)
            - Flexible embedding providers (OpenAI, HuggingFace)
            
            **File Support:**
            - PDF files only (max 50MB each)
            - Multiple file processing
            - Automatic metadata extraction
            - Source attribution and tracking
            - Session-based processing for uploaded files
            
            **Privacy & Performance:**
            - No chat history retention
            - Independent query processing
            - Automatic cleanup of temporary files
            - Session-based resource management
            - Enhanced retrieval with k=8 similarity search
            - Configurable temperature and token limits
            """)
        
        # Advanced configuration
        with st.expander("🔧 Advanced Configuration"):
            st.markdown("**Current Configuration:**")
            if st.session_state.config:
                config_info = f"""
                - **Provider**: {st.session_state.config.model_provider}
                - **Model**: {st.session_state.config.openai_model if st.session_state.config.model_provider == ModelProvider.OPENAI else st.session_state.config.ollama_model}
                - **Embedding Provider**: {st.session_state.config.embedding_provider}
                - **Embedding Model**: {st.session_state.config.embedding_model}
                - **Temperature**: {st.session_state.config.temperature}
                - **Max Tokens**: {st.session_state.config.max_tokens}
                """
                st.markdown(config_info)
            
            # Temperature adjustment
            new_temperature = st.slider(
                "Temperature (creativity vs accuracy):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config.temperature,
                step=0.1,
                help="Lower values = more focused, higher values = more creative"
            )
            
            # Max tokens adjustment
            new_max_tokens = st.slider(
                "Max Response Tokens:",
                min_value=500,
                max_value=4000,
                value=st.session_state.config.max_tokens,
                step=100,
                help="Maximum length of generated responses"
            )
            
            # Update config if values changed
            if (new_temperature != st.session_state.config.temperature or 
                new_max_tokens != st.session_state.config.max_tokens):
                
                st.session_state.config.temperature = new_temperature
                st.session_state.config.max_tokens = new_max_tokens
                
                if st.session_state.system_ready:
                    st.info("⚠️ Configuration changed. Reset system to apply changes.")

if __name__ == "__main__":
    main()