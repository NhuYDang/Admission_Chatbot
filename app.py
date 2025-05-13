import os
import logging
import uuid
import datetime
import json
import re
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "admission-consultant-secret-key")


# Helper function to clean HTML responses
def clean_html_response(text):
    """Clean HTML responses to ensure proper rendering on the frontend"""
    if not text:
        return ""
    
    # Remove Markdown code block markers
    text = re.sub(r'```html', '', text)
    text = re.sub(r'```', '', text)
    
    # Make sure HTML tags are properly formatted
    # Convert &lt; to < and &gt; to > if they exist
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    
    # Make sure all HTML tags are properly closed
    # This is a simple check and won't catch all issues
    common_tags = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'span', 'strong', 'b', 'i', 'em', 'small']
    for tag in common_tags:
        # Count opening and closing tags
        open_count = text.count(f'<{tag}')
        close_count = text.count(f'</{tag}')
        
        # If there are more opening tags than closing tags, add closing tags
        if open_count > close_count:
            text += f'</{tag}>' * (open_count - close_count)
    
    logger.debug(f"Cleaned HTML response. First 100 chars: {text[:100]}")
    return text
    
# PDF files are pre-loaded from the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Import utilities after app is created
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.vector_store_transformers import TransformerVectorStore
from utils.vector_store_tfidf import VectorStore
from utils.gemini_api import generate_response
from utils.orchestrator import orchestrate_response
from utils.conversation_handler import ConversationHandler
import asyncio

# Dictionary to store file information: key = file_id, value = {filename, content} 
file_information = {}

# vector_store = VectorStore()
vector_store = TransformerVectorStore()


# Initialize conversation handler for detecting and responding to conversational queries
conversation_handler = ConversationHandler()
logger.info("Initialized conversation handler for conversational queries")

# Function to load documents from uploaded PDFs
def load_existing_documents():
    if not os.path.exists(UPLOAD_FOLDER):
        return
    
    # Clear existing vector store to avoid duplicate entries
    # vector_store_path = "vector_store_data/tfidf_store.pkl"
    vector_store_path = "vector_store_data/transformer_store.pkl"
    
    if os.path.exists(vector_store_path):
        try:
            if vector_store.load_from_disk(vector_store_path):
                logger.info(f"Loaded vector store from {vector_store_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load vector store from disk: {e}")
            logger.info("Falling back to rebuilding vector store from PDF files")

    # Nếu không có pickle hoặc lỗi, thì build lại từ PDF
        if not os.path.exists(UPLOAD_FOLDER):
            return
        
    vector_store.clear()
    global file_information
    file_information = {}  # Reset file information dictionary
    
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files in uploads folder")
    
    for pdf_file in pdf_files:
        filepath = os.path.join(UPLOAD_FOLDER, pdf_file)
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(filepath)
            
            # Store file information for reference
            file_id = pdf_file
            file_information[file_id] = {
                'filename': pdf_file,
                'content': text
            }
            
            # Chunk text for vector database and tag with source file
            chunks = chunk_text(text, file_source=pdf_file)
            
            # Log sample of chunks to debug
            if chunks:
                # Get sample text (skip the source tag line for logging)
                sample_chunk = chunks[0]
                if sample_chunk.startswith("SOURCE_FILE:"):
                    # Extract content after the source tag
                    sample_lines = sample_chunk.split('\n', 1)
                    if len(sample_lines) > 1:
                        sample_chunk = sample_lines[1]
                
                logger.debug(f"Sample chunk from {pdf_file}: {sample_chunk[:200]}...")
                logger.info(f"Extracted {len(chunks)} chunks from {pdf_file}")
            
            # Add chunks to vector store with source information
            vector_store.add_documents(chunks, file_source=pdf_file)            
            logger.info(f"Processed and loaded {pdf_file} into vector store")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file}: {e}")
    try:
        os.makedirs("vector_store_data", exist_ok=True)
        vector_store.save_to_disk(vector_store_path)
        logger.info(f"Saved vector store to {vector_store_path}")
    except Exception as e:
        logger.error(f"Failed to save vector store to disk: {e}")
        
# Load existing documents on startup
load_existing_documents()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/current_chat_history', methods=['GET'])
def get_current_chat_history():
    return jsonify(session.get('chat_history', []))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get('message', '')
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = []

        chat_history = session['chat_history']
        
        # Check if the query is conversational (greeting, small talk, etc.)
        conversational_response = conversation_handler.get_response(user_query)
        if conversational_response:
            logger.info(f"Detected conversational query, responding with predefined response")
            
            # # Update chat history
            # chat_history.append({"role": "user", "content": user_query})
            # chat_history.append({"role": "assistant", "content": conversational_response})
            
            if not any(msg["content"] == user_query and msg["role"] == "user" for msg in chat_history):
                chat_history.append({"role": "user", "content": user_query})
            if not any(msg["content"] == conversational_response and msg["role"] == "assistant" for msg in chat_history):
                chat_history.append({"role": "assistant", "content": conversational_response})
            
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            session['chat_history'] = chat_history
            
            return jsonify({
                'response': conversational_response,
                'chat_history': chat_history
            })
        
        # Not a conversational query, proceed with retrieval-based response
        # Retrieve relevant context using RAG - find documents related to the user's query
        # For better results, we'll use the similarity search with many results and a low threshold
        # to get as much contextually relevant information as possible
        if vector_store.documents:
            # Get more relevant documents (up to 10) with a very low similarity threshold
            context_docs = vector_store.similarity_search(user_query, k=10, threshold=0.001)
            logger.info(f"Found {len(context_docs)} relevant documents for query: {user_query}")
            
            # If no documents found with similarity search, use all documents
            if not context_docs or len(context_docs) == 0 or len(context_docs[0]) < 100:
                # Use all documents but limit total context size
                max_docs = min(5, len(vector_store.documents))  # Use up to 5 documents
                context_docs = vector_store.documents[:max_docs]
                logger.info(f"Using {len(context_docs)} full documents as context for query: {user_query}")
        else:
            # No documents available
            context_docs = ["No documents available in the knowledge base yet. Please upload PDF files."]
        
        # Extract file sources from context documents
        file_sources = []
        processed_docs = []
        
        for doc in context_docs:
            # Parse source file information from the document
            try:
                if doc.startswith("SOURCE_FILE:"):
                    # Extract the source file name from the tag
                    source_line = doc.split('\n')[0]  # Get the first line with source info
                    file_id = source_line.replace("SOURCE_FILE:", "").strip()
                    
                    # Add to list of sources if not already included
                    if file_id not in file_sources:
                        file_sources.append(file_id)
                    
                    # Remove the source tag line for the content
                    processed_content = "\n".join(doc.split('\n')[1:])
                    processed_docs.append(processed_content)
                else:
                    # No source tag, use as is
                    processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error parsing document source: {e}")
                processed_docs.append(doc)  # Use original if any error
        
        # Convert file IDs to original filenames for better human readability
        readable_sources = []
        for file_id in file_sources:
            if file_id in file_information:
                # Use the original filename stored during upload
                readable_sources.append(file_information[file_id]['filename'])
            else:
                # For files with UUID prefixes, extract the original name
                if '_' in file_id and any(c for c in file_id if c == '-'):
                    # This is likely a UUID prefixed filename pattern
                    try:
                        # Extract the part after UUID
                        original_name = file_id.split('_', 1)[1] if '_' in file_id else file_id
                        readable_sources.append(original_name)
                    except:
                        # Fallback to the ID if extraction fails
                        readable_sources.append(file_id)
                else:
                    # Just use the filename directly if no UUID pattern
                    readable_sources.append(file_id)
        
        # Log context found for debugging
        if processed_docs:
            logger.info(f"Found {len(processed_docs)} relevant context documents from {len(file_sources)} files for query: {user_query}")
            logger.info(f"Source files: {', '.join(readable_sources)}")
            for i, doc in enumerate(processed_docs):
                logger.debug(f"Context {i+1}: {doc[:200]}...")
                
        context = "\n\n".join(processed_docs)
        
        # Add detailed file sources to the context for the API with file prioritization hints
        if readable_sources:
            # Group files by type to help with prioritization hints
            file_categories = {
                'diem_chuan': [],
                'hoc_phi': [],
                'nganh_hoc': [],
                'co_so': [],
                'tuyen_sinh': [],
                'other': []
            }
            
            # Categorize files for better searching
            for filename in readable_sources:
                filename_lower = filename.lower()
                if 'diem' in filename_lower or 'chuan' in filename_lower:
                    file_categories['diem_chuan'].append(filename)
                elif 'hoc_phi' in filename_lower or 'phi' in filename_lower:
                    file_categories['hoc_phi'].append(filename)
                elif 'nganh' in filename_lower or 'khoa' in filename_lower:
                    file_categories['nganh_hoc'].append(filename)
                elif 'co_so' in filename_lower or 'vat_chat' in filename_lower:
                    file_categories['co_so'].append(filename)
                elif 'tuyen_sinh' in filename_lower or 'tuyen' in filename_lower:
                    file_categories['tuyen_sinh'].append(filename)
                else:
                    file_categories['other'].append(filename)
            
            # Add metadata about files and prioritization hints
            source_info = "\n\n### SOURCE FILES AND PRIORITIZATION HINTS:\n"
            source_info += "THIS INFORMATION IS FOUND IN THE FOLLOWING FILES: " + ", ".join(readable_sources) + "\n"
            
            # Add hints about file categories to help API prioritize correctly
            if file_categories['diem_chuan']:
                source_info += "\nFILES ABOUT ĐIỂM CHUẨN (PRIORITY FOR QUESTIONS ABOUT ADMISSION SCORES): " + ", ".join(file_categories['diem_chuan'])
            if file_categories['hoc_phi']:
                source_info += "\nFILES ABOUT HỌC PHÍ (PRIORITY FOR QUESTIONS ABOUT TUITION FEES): " + ", ".join(file_categories['hoc_phi'])
            if file_categories['nganh_hoc']:
                source_info += "\nFILES ABOUT NGÀNH HỌC (PRIORITY FOR QUESTIONS ABOUT MAJORS/DEPARTMENTS): " + ", ".join(file_categories['nganh_hoc'])
            if file_categories['co_so']:
                source_info += "\nFILES ABOUT CƠ SỞ VẬT CHẤT (PRIORITY FOR QUESTIONS ABOUT FACILITIES): " + ", ".join(file_categories['co_so'])
            if file_categories['tuyen_sinh']:
                source_info += "\nFILES ABOUT TUYỂN SINH (PRIORITY FOR QUESTIONS ABOUT ADMISSIONS): " + ", ".join(file_categories['tuyen_sinh'])
            if file_categories['other']:
                source_info += "\nOTHER FILES: " + ", ".join(file_categories['other'])
                
            context += source_info
        
        # Debug: Log length of context
        logger.info(f"Context length: {len(context)} characters")
        
        # Use Orchestrator-workers model for better search and extraction
        # Run the async orchestrator in a synchronous environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                orchestrate_response(user_query, processed_docs, file_sources)
            )
        finally:
            loop.close()
            
        # Fallback to regular Gemini if orchestrator fails or returns None
        if not response or (isinstance(response, str) and "error" in response.lower()):
            logger.warning(f"Orchestrator failed, falling back to standard Gemini API")
            try:
                response = generate_response(user_query, context, chat_history)
            except Exception as gemini_error:
                logger.error(f"Error with Gemini API: {gemini_error}")
                # Provide a fallback response when both methods fail
                response = """
                <div class="alert alert-warning">
                    <h4>Không tìm thấy thông tin</h4>
                    <p>Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu hiện có.</p>
                    <p>Câu hỏi của bạn có thể nằm ngoài phạm vi thông tin tuyển sinh hoặc các tài liệu đã tải lên. 
                    Vui lòng thử lại với một câu hỏi khác về thông tin tuyển sinh, hoặc liên hệ với phòng tuyển sinh để được hỗ trợ thêm.</p>
                </div>
                """
        
        # Clean the response HTML to ensure proper rendering
        response = clean_html_response(response)

        # # Update chat history
        # chat_history.append({"role": "user", "content": user_query})
        # chat_history.append({"role": "assistant", "content": response})

        # Only append if not already in history
        if not any(msg["content"] == user_query and msg["role"] == "user" for msg in chat_history):
            chat_history.append({"role": "user", "content": user_query})
        if not any(msg["content"] == response and msg["role"] == "assistant" for msg in chat_history):
            chat_history.append({"role": "assistant", "content": response})
        
        # Limit history length to prevent session from getting too large
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        session['chat_history'] = chat_history
        logger.debug(f"Returning chat history: {chat_history}")  # Debug log

        return jsonify({
            'response': response,
            'chat_history': chat_history
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Clear chat history from flask session
    session.pop('chat_history', None)
    return jsonify({'stat us': 'success'})

def calculate_relevance_score(query, documents):
    """Calculate a relevance score for documents based on keyword matching with the query
    
    This is a simple heuristic - a real system would use human evaluation or more sophisticated metrics
    """
    if not documents:
        return 0
    
    # Extract keywords from query (simple approach - split on spaces and filter)
    query = query.lower()
    query_words = set([w for w in query.split() if len(w) > 3])
    
    # If no significant query words, consider any word
    if not query_words:
        query_words = set(query.split())
    
    # Count relevant documents (contain at least one query keyword)
    relevant_count = 0
    
    for doc in documents:
        # Extract text content based on document type
        if hasattr(doc, 'page_content'):
            # Langchain document
            doc_text = doc.page_content
        elif isinstance(doc, str):
            # String document
            doc_text = doc
            # Remove source information if present
            if doc.startswith("SOURCE_FILE:"):
                parts = doc.split("\n", 1)
                if len(parts) > 1:
                    doc_text = parts[1]
        elif isinstance(doc, dict):
            # Dictionary document
            doc_text = doc.get('content', str(doc))
        else:
            # Fallback
            doc_text = str(doc)
        
        # Convert to lowercase for case-insensitive matching
        doc_lower = doc_text.lower()
        
        # Count matching keywords
        matching_keywords = sum(1 for word in query_words if word in doc_lower)
        
        # If the document contains any keywords or matches the query semantically, consider it relevant
        if matching_keywords > 0:
            # Look for increasing relevance based on keyword density
            if matching_keywords >= len(query_words) * 0.2 or query.lower() in doc_lower:
                relevant_count += 1
    
    # Calculate precision
    if documents:
        return relevant_count / len(documents)
    else:
        return 0

if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)