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
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "admission-consultant-secret-key")

# Database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
else:
    logger.error("DATABASE_URL environment variable not found")
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://neondb_owner:npg_t0jHNzSx7Osg@ep-winter-cherry-a6q473sj.us-west-2.aws.neon.tech/neondb?sslmode=require"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

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
from utils.vector_store import VectorStore
from utils.gemini_api import generate_response
from utils.orchestrator import orchestrate_response
from utils.conversation_handler import ConversationHandler
import asyncio

# Dictionary to store file information: key = file_id, value = {filename, content} 
file_information = {}

# Import models and create tables
with app.app_context():
    import models
    db.create_all()

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

# Load existing documents on startup
load_existing_documents()

# Helper function to update database with chat messages
def save_chat_message(session_id, role, content):
    """Save a chat message to the database"""
    try:
        from models import ChatMessage
        
        new_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        db.session.add(new_message)
        db.session.commit()
        logger.debug(f"Saved {role} message to database, session_id: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving message to database: {e}")
        return False


@app.route('/')
def index():
    return render_template('index.html')

# Chat sessions API endpoints
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions for display in sidebar"""
    try:
        from models import ChatSession
        sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).all()
        result = []
        for session in sessions:
            result.append({
                'id': session.id,
                'session_id': session.session_id,
                'title': session.title or f"Chat {session.id}",
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat()
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new chat session"""
    try:
        from models import ChatSession
        new_session = ChatSession(
            session_id=str(uuid.uuid4()),
            title=f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        db.session.add(new_session)
        db.session.commit()
        
        # Clear session chat history for new chat
        session['chat_history'] = []
        session['current_session_id'] = new_session.id
        
        return jsonify({
            'id': new_session.id,
            'session_id': new_session.session_id,
            'title': new_session.title,
            'created_at': new_session.created_at.isoformat(),
            'updated_at': new_session.updated_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific chat session with messages"""
    try:
        from models import ChatSession, ChatMessage
        chat_session = ChatSession.query.get_or_404(session_id)
        messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at).all()
        
        # Set current session in flask session
        session['current_session_id'] = session_id
        
        # Build chat history for this session
        chat_history = []
        for msg in messages:
            chat_history.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # Store in session for continuation
        session['chat_history'] = chat_history
        
        # Format messages for response, cleaning HTML content in assistant messages
        formatted_messages = []
        for msg in messages:
            content = msg.content
            # Clean HTML in assistant messages
            if msg.role == 'assistant':
                content = clean_html_response(content)
                
            formatted_messages.append({
                'id': msg.id,
                'role': msg.role,
                'content': content,
                'created_at': msg.created_at.isoformat()
            })
        
        return jsonify({
            'session': {
                'id': chat_session.id,
                'session_id': chat_session.session_id,
                'title': chat_session.title,
                'created_at': chat_session.created_at.isoformat(),
                'updated_at': chat_session.updated_at.isoformat()
            },
            'messages': formatted_messages
        })
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get('message', '')
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Check if the query is conversational (greeting, small talk, etc.)
        conversational_response = conversation_handler.get_response(user_query)
        if conversational_response:
            logger.info(f"Detected conversational query, responding with predefined response")
            
            # Get current session ID or create a new one
            current_session_id = session.get('current_session_id')
            if not current_session_id:
                # Create a new session if none exists
                from models import ChatSession
                new_session = ChatSession(
                    session_id=str(uuid.uuid4()),
                    title=f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                db.session.add(new_session)
                db.session.commit()
                current_session_id = new_session.id
                session['current_session_id'] = current_session_id
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": conversational_response})
            
            # Save messages to database
            save_chat_message(current_session_id, "user", user_query)
            save_chat_message(current_session_id, "assistant", conversational_response)
            
            # Update the session updated_at timestamp
            from models import ChatSession
            chat_session = ChatSession.query.get(current_session_id)
            if chat_session:
                chat_session.updated_at = datetime.datetime.utcnow()
                db.session.commit()
            
            # Limit history length to prevent session from getting too large
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            
            session['chat_history'] = chat_history
            
            return jsonify({
                'response': conversational_response
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
        
        # Get current session ID
        current_session_id = session.get('current_session_id')
        if not current_session_id:
            # Create a new session if none exists
            from models import ChatSession
            new_session = ChatSession(
                session_id=str(uuid.uuid4()),
                title=f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            db.session.add(new_session)
            db.session.commit()
            current_session_id = new_session.id
            session['current_session_id'] = current_session_id
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response})
        
        # Save messages to database
        save_chat_message(current_session_id, "user", user_query)
        save_chat_message(current_session_id, "assistant", response)
        
        # Update the session updated_at timestamp
        from models import ChatSession
        chat_session = ChatSession.query.get(current_session_id)
        if chat_session:
            chat_session.updated_at = datetime.datetime.utcnow()
            db.session.commit()
        
        # Limit history length to prevent session from getting too large
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': response
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Clear chat history from flask session
    session.pop('chat_history', None)
    
    # Clear messages from database for current session
    current_session_id = session.get('current_session_id')
    if current_session_id:
        try:
            from models import ChatMessage
            # Delete all messages for this session
            ChatMessage.query.filter_by(session_id=current_session_id).delete()
            db.session.commit()
            logger.info(f"Cleared all messages for session {current_session_id}")
        except Exception as e:
            logger.error(f"Error clearing messages from database: {e}")
    
    return jsonify({'status': 'success'})


@app.route('/api/embedding/benchmark', methods=['POST'])
def benchmark_embeddings():
    """API endpoint to benchmark different embedding methods with actual evaluation metrics"""
    try:
        data = request.json
        test_queries = data.get('queries', [])
        current_embedding = data.get('current_embedding', 'tfidf')
        timing_iterations = data.get('timing_iterations', 3)  # Default to 3 iterations for timing
        
        if not test_queries:
            return jsonify({'error': 'No test queries provided'}), 400
        
        # Create temporary vector stores for evaluation
        tfidf_store = VectorStoreFactory.create_vector_store(store_type=VectorStoreFactory.TFIDF)
        transformer_store = VectorStoreFactory.create_vector_store(store_type=VectorStoreFactory.TRANSFORMER)
        
        # Load the same documents into both stores
        # Just load a subset of documents to speed up the benchmark process
        for pdf_file in os.listdir(UPLOAD_FOLDER)[:5]:  # Limit to first 5 PDFs for faster testing
            if pdf_file.endswith('.pdf'):
                filepath = os.path.join(UPLOAD_FOLDER, pdf_file)
                try:
                    text = extract_text_from_pdf(filepath)
                    chunks = chunk_text(text, file_source=pdf_file)
                    
                    # Add to both vector stores
                    tfidf_store.add_documents(chunks, file_source=pdf_file)
                    transformer_store.add_documents(chunks, file_source=pdf_file)
                except Exception as e:
                    logger.error(f"Error loading {pdf_file} for benchmark: {e}")
        
        # Results dictionary to store metrics
        results = {
            'tfidf': {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'response_time': 0
            },
            'transformer': {
                'precision': 0,
                'recall': 0, 
                'f1_score': 0,
                'response_time': 0
            }
        }
        
        # Perform benchmark for each vector store type
        for store_type in ['tfidf', 'transformer']:
            store = tfidf_store if store_type == 'tfidf' else transformer_store
            
            # Track metrics across all test queries
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_time = 0
            
            for query in test_queries:
                # For timing accuracy, run the query multiple times and take the average
                query_times = []
                precision_values = []
                
                for _ in range(timing_iterations):
                    # Measure response time
                    start_time = datetime.datetime.now()
                    
                    # Get relevant documents with a fixed number of results
                    relevant_docs = store.similarity_search(query, k=5, threshold=0.001)
                    
                    # Simulate full processing time including post-processing that would happen in a real query
                    # This better represents the actual user experience time
                    processed_docs = []
                    for doc in relevant_docs:
                        # Add proper processing time simulation with some real work
                        # Handle different document formats that might be returned from different vector stores
                        if hasattr(doc, 'page_content'):
                            # This is a Langchain document format
                            processed_content = doc.page_content
                            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                        elif isinstance(doc, str):
                            # This is a plain string format
                            processed_content = doc
                            # Extract source if string has format like "SOURCE_FILE:filename.pdf\nContent..."
                            if doc.startswith("SOURCE_FILE:"):
                                parts = doc.split("\n", 1)
                                source = parts[0].replace("SOURCE_FILE:", "").strip()
                                processed_content = parts[1] if len(parts) > 1 else doc
                            else:
                                source = 'Unknown'
                        elif isinstance(doc, dict):
                            # This is a dictionary format
                            processed_content = doc.get('content', str(doc))
                            source = doc.get('source', 'Unknown')
                        else:
                            # Fallback for any other format
                            processed_content = str(doc)
                            source = 'Unknown'
                        
                        # Perform actual text processing to simulate real workload
                        # Split into sentences, count words, find keywords - these operations take time
                        sentences = processed_content.split('.')
                        word_count = len(processed_content.split())
                        keywords = [word for word in query.lower().split() if word in processed_content.lower()]
                        
                        # Create a processed document with all extracted information
                        processed_docs.append({
                            'content': processed_content,
                            'source': source,
                            'sentences': len(sentences),
                            'word_count': word_count,
                            'keyword_matches': len(keywords)
                        })
                    
                    # Calculate response time including processing
                    end_time = datetime.datetime.now()
                    iteration_time = (end_time - start_time).total_seconds()
                    query_times.append(iteration_time)
                    
                    # Log the timing for debugging
                    logger.debug(f"Query timing for {store_type}, iteration {_+1}: {iteration_time:.4f}s")
                    
                    # Calculate precision for this iteration
                    precision = calculate_relevance_score(query, relevant_docs)
                    precision_values.append(precision)
                    
                    # Add a small delay between iterations to prevent CPU caching effects
                    # and to give more consistent timing results
                    time.sleep(0.1)  # 100ms delay
                
                # Use the average time from all iterations for more accurate timing
                avg_query_time = sum(query_times) / len(query_times)
                total_time += avg_query_time
                
                # Use the average precision from all iterations
                avg_precision = sum(precision_values) / len(precision_values)
                total_precision += avg_precision
                
                # For this demo, we'll use simplified metrics since we don't have labeled ground truth
                # In a real system, you would compare against known relevant documents
                # Here we'll set recall equal to precision as an approximation
                recall = avg_precision
                total_recall += recall
                
                # Calculate F1 score
                if avg_precision + recall > 0:
                    f1 = 2 * (avg_precision * recall) / (avg_precision + recall)
                else:
                    f1 = 0
                    
                total_f1 += f1
            
            # Calculate averages
            if len(test_queries) > 0:
                results[store_type]['precision'] = round((total_precision / len(test_queries)) * 100)
                results[store_type]['recall'] = round((total_recall / len(test_queries)) * 100)
                results[store_type]['f1_score'] = round((total_f1 / len(test_queries)) * 100)
                results[store_type]['response_time'] = round(total_time / len(test_queries), 2)
        
        # Return only the requested embedding type if specified
        if current_embedding in results:
            return jsonify({'results': results[current_embedding], 'total_queries': len(test_queries)})
        else:
            return jsonify({'results': results, 'total_queries': len(test_queries)})
            
    except Exception as e:
        logger.error(f"Error in benchmark_embeddings: {e}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

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