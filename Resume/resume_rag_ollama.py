import json
import pandas as pd
from pathlib import Path
import os
import tempfile
import shutil
from typing import List, Optional, Tuple, Dict, Any
import hashlib
from dotenv import load_dotenv

load_dotenv()

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings as HuggingFaceEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

PDF_FOLDER = "Resume"  
DB_FOLDER = "./resume_db"  
OLLAMA_MODEL_NAME = "gemma3:1b"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
COLLECTION_NAME = "resume-collection"
EMBEDDING_MODEL = "thenlper/gte-large"
TEMP_DB_PREFIX = "temp_resume_db_"

class ModelProvider:
    OLLAMA = "ollama"
    OPENAI = "openai"

class RAGConfig:
    def __init__(self, 
                 model_provider: str = ModelProvider.OPENAI,
                 ollama_model: str = OLLAMA_MODEL_NAME,
                 openai_model: str = OPENAI_MODEL_NAME,
                 embedding_provider: str = "huggingface",
                 embedding_model: str = EMBEDDING_MODEL,
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        self.model_provider = model_provider
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens

def validate_environment():
    openai_key = os.getenv("OPENAI_API_KEY")
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_project = os.getenv("LANGCHAIN_PROJECT")
    
    if langchain_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_key
    if langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
    return {
        "openai_available": bool(openai_key),
        "langchain_available": bool(langchain_key),
        "langchain_project": langchain_project
    }

def create_pdf_hash(pdf_files: List) -> str:
    if not pdf_files:
        return "empty"
    
    hash_string = ""
    for pdf_file in pdf_files:
        if hasattr(pdf_file, 'name') and hasattr(pdf_file, 'size'):
            hash_string += f"{pdf_file.name}_{pdf_file.size}_"
    
    return hashlib.md5(hash_string.encode()).hexdigest()[:10]

def save_uploaded_pdfs(uploaded_files: List, temp_dir: str) -> List[str]:
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(file_path)
    
    return saved_paths

def process_uploaded_pdfs(uploaded_files: List) -> Tuple[Optional[str], Optional[List[Document]], Optional[List[str]]]:
    if not uploaded_files:
        return None, None, None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix=TEMP_DB_PREFIX)
        saved_paths = save_uploaded_pdfs(uploaded_files, temp_dir)
        
        documents = []
        filenames = []
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap=16
        )
        
        for file_path in saved_paths:
            filename = os.path.basename(file_path)
            filenames.append(filename)
            loader = PyPDFLoader(file_path)
            file_docs = loader.load_and_split(text_splitter)
            
            for doc in file_docs:
                doc.metadata['source_filename'] = filename
                doc.metadata['original_source'] = filename
            
            documents.extend(file_docs)
        
        return temp_dir, documents, filenames
        
    except Exception as e:
        return None, None, None

def create_embedding_model(config: RAGConfig):
    try:
        if config.embedding_provider == "openai" and OpenAIEmbeddings:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found in environment")
            
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
        else:
            try:
                embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
            except Exception as e:
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        return embedding_model
        
    except Exception as e:
        return None

def create_dynamic_vectorstore(documents: List[Document], session_id: str, config: RAGConfig) -> Optional[Chroma]:
    if not documents:
        return None
    
    try:
        embedding_model = create_embedding_model(config)
        if not embedding_model:
            return None
        
        temp_db_dir = f"./temp_dbs/session_{session_id}"
        os.makedirs(temp_db_dir, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents,
            embedding_model,
            collection_name=f"session_{session_id}",
            persist_directory=temp_db_dir
        )
        vectorstore.persist()
        
        return vectorstore
        
    except Exception as e:
        return None

def create_vector_database(pdf_loader, text_splitter, config: RAGConfig):
    try:
        resume_chunks = pdf_loader.load_and_split(text_splitter)
        
        embedding_model = create_embedding_model(config)
        if not embedding_model:
            return None, None
        
        vectorstore = Chroma.from_documents(
            resume_chunks,
            embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_FOLDER
        )
        vectorstore.persist()
        
        return embedding_model, vectorstore
        
    except Exception as e:
        return None, None

def load_vector_database(config: RAGConfig):
    try:
        embedding_model = create_embedding_model(config)
        if not embedding_model:
            return None, None
        
        vectorstore_persisted = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_FOLDER,
            embedding_function=embedding_model
        )
        
        return embedding_model, vectorstore_persisted
        
    except Exception as e:
        return None, None

def initialize_llm(config: RAGConfig):
    try:
        if config.model_provider == ModelProvider.OPENAI:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found in environment")
            
            llm = ChatOpenAI(
                model=config.openai_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            test_response = llm.invoke([HumanMessage(content="Hello")])
            
        else:
            llm = Ollama(
                model=config.ollama_model,
                temperature=config.temperature
            )
            
            test_response = llm.invoke("Hello")
        
        return llm
        
    except Exception as e:
        return None

def setup_document_processing():
    if not Path(PDF_FOLDER).exists():
        return None, None
    
    pdf_loader = PyPDFDirectoryLoader(PDF_FOLDER)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=512,
        chunk_overlap=16
    )
    
    return pdf_loader, text_splitter

def cleanup_temp_files(temp_dir: str, temp_db_dir: str = None):
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        if temp_db_dir and os.path.exists(temp_db_dir):
            shutil.rmtree(temp_db_dir)
            
    except Exception as e:
        pass

qna_system_message = """
You are an expert document analysis assistant with advanced capabilities in extracting, synthesizing, and presenting information from PDF documents. Your primary functions include comprehensive document summarization, precise question answering, and educational content generation.

CORE COMPETENCIES:
- Advanced text comprehension and synthesis
- Multi-document comparative analysis
- Structured information extraction
- Educational content creation
- Professional document assessment

OPERATIONAL GUIDELINES:

1. **ENHANCED SUMMARIZATION MODE**
   When users request summaries, provide in-depth, professionally structured analysis:
   
   For EACH document, deliver:
   ### Document: [ORIGINAL_FILENAME]
   
   **Executive Summary:** [2-3 sentence overview]
   **Key Highlights:** [Main points, achievements, qualifications]
   **Technical Details:** [Specific skills, technologies, certifications]
   **Professional Background:** [Experience levels, industries, roles]
   **Notable Elements:** [Unique aspects, standout features]
   **Assessment:** [Professional evaluation of strengths]
   
   Requirements:
   - Extract ALL significant information from each document
   - Maintain professional, analytical tone
   - Provide actionable insights
   - Ensure completeness and accuracy
   - Use clear, structured formatting

2. **PRECISION QUESTION ANSWERING**
   Deliver comprehensive, evidence-based responses:
   - Analyze ALL relevant context thoroughly
   - Provide detailed, multi-faceted answers
   - Include specific examples and evidence
   - Cross-reference information when applicable
   - Maintain factual accuracy and completeness
   - Structure complex answers with clear organization

3. **ADVANCED QUIZ GENERATION**
   Create sophisticated assessment materials:
   - Generate minimum 12-15 high-quality questions
   - Employ diverse question types:
     * Multiple choice (4 options, single correct)
     * True/False with justification requirements
     * Short answer (2-3 sentences expected)
     * Fill-in-the-blank with context
     * Scenario-based application questions
     * Comparative analysis questions
   - Ensure comprehensive content coverage
   - Include questions of varying difficulty levels
   - Focus on practical application and understanding

4. **MULTI-REQUEST PROCESSING**
   Handle complex, multi-part requests systematically:
   - Parse all request components accurately
   - Execute each task with full attention to detail
   - Provide clear section delineation
   - Maintain consistent quality across all deliverables

QUALITY STANDARDS:
- Professional, clear, and engaging communication
- Comprehensive coverage of all available information
- Accurate representation of document contents
- Structured, logical presentation
- Evidence-based analysis and recommendations

CONSTRAINTS:
- Rely exclusively on provided context information
- Never fabricate or assume information not present
- Maintain objectivity and factual accuracy
- Preserve original document integrity and meaning

Your responses should demonstrate expertise, thoroughness, and professional insight while remaining accessible and actionable for the user.
"""

qna_user_message_template = """
###Context
Here are the relevant document sections for analysis:
{context}

###Question
{question}
"""

def get_original_filenames_from_context(relevant_documents: List[Document]) -> List[str]:
    filenames = set()
    for doc in relevant_documents:
        if 'source_filename' in doc.metadata:
            filenames.add(doc.metadata['source_filename'])
        elif 'original_source' in doc.metadata:
            filenames.add(doc.metadata['original_source'])
        elif 'source' in doc.metadata:
            source_path = doc.metadata['source']
            filename = os.path.basename(source_path)
            filenames.add(filename)
    return list(filenames)

def run_qna_pipeline(user_input: str, retriever, llm, config: RAGConfig) -> str:
    try:
        relevant_document_chunks = retriever.get_relevant_documents(user_input)
        
        filenames = get_original_filenames_from_context(relevant_document_chunks)
        
        context_list = []
        for doc in relevant_document_chunks:
            filename = "Unknown"
            if 'source_filename' in doc.metadata:
                filename = doc.metadata['source_filename']
            elif 'original_source' in doc.metadata:
                filename = doc.metadata['original_source']
            elif 'source' in doc.metadata:
                filename = os.path.basename(doc.metadata['source'])
            
            context_entry = f"[Source: {filename}]\n{doc.page_content}"
            context_list.append(context_entry)
        
        context_for_query = "\n\n".join(context_list)
        
        if "summary" in user_input.lower() or "summarize" in user_input.lower():
            enhanced_context = f"Available documents for analysis: {', '.join(filenames)}\n\n{context_for_query}"
        else:
            enhanced_context = context_for_query

        if config.model_provider == ModelProvider.OPENAI:
            messages = [
                SystemMessage(content=qna_system_message),
                HumanMessage(content=qna_user_message_template.format(
                    context=enhanced_context, 
                    question=user_input
                ))
            ]
            response = llm.invoke(messages)
            return response.content.strip()
        else:
            full_prompt = f"{qna_system_message}\n\n{qna_user_message_template.format(context=enhanced_context, question=user_input)}"
            response = llm.invoke(full_prompt)
            return response.strip()

    except Exception as e:
        return f"Error: {e}"

def setup_dynamic_system(uploaded_files: List = None, 
                        session_id: str = "default", 
                        config: RAGConfig = None) -> Tuple[Any, Any, Any, str, List[str]]:
    if config is None:
        config = RAGConfig()
    
    env_status = validate_environment()
    
    vectorstore_persisted = None
    temp_dir = None
    filenames = None
    
    if uploaded_files:
        temp_dir, documents, filenames = process_uploaded_pdfs(uploaded_files)
        if documents:
            vectorstore_persisted = create_dynamic_vectorstore(documents, session_id, config)
        else:
            return None, None, None, None, None
    else:
        db_exists = Path(DB_FOLDER).exists()
        
        if db_exists:
            embedding_model, vectorstore_persisted = load_vector_database(config)
        else:
            pdf_loader, text_splitter = setup_document_processing()
            if pdf_loader is None:
                return None, None, None, None, None
            
            embedding_model, vectorstore_persisted = create_vector_database(pdf_loader, text_splitter, config)
    
    if vectorstore_persisted is None:
        return None, None, None, None, None
    
    llm = initialize_llm(config)
    if llm is None:
        return None, None, None, None, None
    
    retriever = vectorstore_persisted.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 8}
    )
    
    return llm, retriever, vectorstore_persisted, temp_dir, filenames

def setup_system(config: RAGConfig = None):
    if config is None:
        config = RAGConfig()
    
    llm, retriever, vectorstore_persisted, _, _ = setup_dynamic_system(config=config)
    return llm, retriever, vectorstore_persisted, config

def create_openai_config(model: str = "gpt-3.5-turbo", temperature: float = 0.1) -> RAGConfig:
    return RAGConfig(
        model_provider=ModelProvider.OPENAI,
        openai_model=model,
        embedding_provider="openai",
        temperature=temperature
    )

def create_ollama_config(model: str = "gemma3:1b", temperature: float = 0.1) -> RAGConfig:
    return RAGConfig(
        model_provider=ModelProvider.OLLAMA,
        ollama_model=model,
        embedding_provider="huggingface",
        temperature=temperature
    )

if __name__ == "__main__":
    env_status = validate_environment()
    
    if env_status["openai_available"]:
        config = create_openai_config()
    else:
        config = create_ollama_config()
    
    llm, retriever, vectorstore_persisted, config = setup_system(config)
    
    if llm is None:
        exit(1)

try:
    if 'llm' not in globals():
        llm, retriever, vectorstore_persisted, config = setup_system()
except Exception as e:
    llm = retriever = vectorstore_persisted = config = None