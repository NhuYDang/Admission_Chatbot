import os
import json
import re
import asyncio
import time
import logging
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .gemini_api import GeminiAgent
from .gemini_api import prepare_vietnamese_context

logger = logging.getLogger(__name__)

class Task:
    """Represents a task to be executed by a worker"""
    def __init__(self, task_id: str, query: str, context: str, source_file: str):
        self.task_id = task_id
        self.query = query
        self.context = context
        self.source_file = source_file
        self.created_at = datetime.now()
        self.completed_at = None
        self.result = None
        self.status = "pending"  # pending, processing, completed, failed
        
    def mark_processing(self):
        self.status = "processing"
        return self
        
    def mark_completed(self, result):
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.now()
        return self
        
    def mark_failed(self, error):
        self.status = "failed"
        self.result = {"error": str(error)}
        self.completed_at = datetime.now()
        return self
        
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "query": self.query,
            "source_file": self.source_file,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "result": self.result,
        }

class Worker:
    """Worker that processes a task by using the Gemini API"""
    def __init__(self, worker_id: str, api_key: Optional[str] = None):
        self.worker_id = worker_id
        self.api_key = api_key
        self.agent = GeminiAgent(api_key=api_key)
        self.processed_tasks = 0
        
    async def process_task(self, task: Task):
        """Process a task asynchronously"""
        logger.info(f"Worker {self.worker_id} processing task {task.task_id} from file {task.source_file}")
        task.mark_processing()
        
        try:
            # Create a focused query for this specific document
            prompt = self._create_worker_prompt(task.query, task.context, task.source_file)
            
            # Execute in thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, self._execute_query, prompt
                )
            
            # Process and structure the result
            task.mark_completed({
                "content": result,
                "source_file": task.source_file,
                "relevance_score": self._calculate_relevance(result, task.query)
            })
            
            self.processed_tasks += 1
            logger.info(f"Worker {self.worker_id} completed task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed on task {task.task_id}: {str(e)}")
            task.mark_failed(e)
            
        return task
    
    def _execute_query(self, prompt: str) -> str:
        """Execute a query using the Gemini API"""
        # Using the agent's _call_api method directly for more control
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }
        
        try:
            response_json = self.agent._call_api(payload)
            return self.agent._process_response(response_json) or ""
        except Exception as e:
            logger.error(f"API call error in worker {self.worker_id}: {e}")
            raise
    
    def _create_worker_prompt(self, query: str, context: str, source_file: str) -> str:
        """Create a prompt for the worker focusing on extracting relevant information"""
        file_type = self._determine_file_type(source_file)
        
        return f"""
        Bạn là trợ lý AI chuyên nghiệp hỗ trợ tư vấn tuyển sinh cho trường Đại học Mở Thành phố Hồ Chí Minh.
        Bạn có khả năng tìm kiếm và trích xuất thông tin từ tài liệu từ đó đưa ra câu trả lời cho người dùng.
        Nhiệm vụ của bạn là đọc tài liệu sau đây và trích xuất thông tin cụ thể liên quan đến câu hỏi.
        Dựa trên câu hỏi và thông tin tìm được, trả lời ngắn gọn, chính xác bằng tiếng Việt. Nếu câu hỏi mơ hồ, hỏi lại để làm rõ.
        Nếu câu hỏi không liên quan đến tư vấn tuyển sinh như những câu chào hỏi, trò chuyện, hãy trả lời một cách thân thiện và vui vẻ.
        
        CÂU HỎI: "{query}"
        
        {context}
        
        HƯỚNG DẪN TRÍCH XUẤT:
        1. Chỉ trích xuất thông tin LIÊN QUAN TRỰC TIẾP đến câu hỏi
        2. Nếu tài liệu KHÔNG chứa thông tin liên quan, hãy trả lời: "Không tìm thấy thông tin liên quan trong tài liệu này."
        3. KHÔNG bịa đặt hoặc suy luận thông tin không có trong tài liệu
        4. Giữ nguyên các con số, tên riêng, và thuật ngữ chuyên ngành
        5. Trích dẫn nội dung quan trọng bằng dấu ngoặc kép
        6. Định dạng thông tin rõ ràng với tiêu đề và cấu trúc
        
        """
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine the type of file based on its name"""
        filename = filename.lower()
        
        if "diem_chuan" in filename:
            return "điểm_chuẩn"
        elif "hoc_phi" in filename or "hoc_bong" in filename:
            return "học_phí_học_bổng"
        elif "nganh_hoc" in filename or "khoa" in filename:
            return "ngành_học"
        elif "co_so_vat_chat" in filename:
            return "cơ_sở_vật_chất"
        elif "tuyen_sinh" in filename:
            return "tuyển_sinh"
        else:
            return "khác"
    
    def _calculate_relevance(self, result: str, query: str) -> float:
        """Calculate a relevance score for the result based on the query"""
        if "Không tìm thấy thông tin liên quan" in result:
            return 0.0
            
        # Simple relevance calculation based on matching keywords
        relevance = 0.0
        
        # Extract keywords from query
        query_words = set(re.findall(r'\w+', query.lower()))
        # Remove common Vietnamese stopwords
        stopwords = {
            'và', 'của', 'là', 'cho', 'trong', 'về', 'từ', 'với', 'đến', 'tại',
            'một', 'các', 'những', 'này', 'đó', 'nên', 'khi', 'thì', 'được',
            'bằng', 'có', 'đã', 'sẽ', 'còn', 'vẫn'
        }
        query_words = query_words - stopwords
        
        # Count matching keywords in result
        result_lower = result.lower()
        matches = sum(1 for word in query_words if word in result_lower)
        
        # Basic relevance score calculation
        if query_words:
            relevance = matches / len(query_words)
        
        # Boost score if result contains numbers (likely factual information)
        if re.search(r'\d+', result):
            relevance *= 1.2
            
        # Cap at 1.0
        return min(relevance, 1.0)

class Orchestrator:
    """Orchestrator that coordinates tasks and workers"""
    def __init__(self, api_key: Optional[str] = None, num_workers: int = 3):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.tasks: List[Task] = []
        self.task_results: Dict[str, Any] = {}
        self.workers: List[Worker] = [
            Worker(f"worker-{i}", api_key=self.api_key) for i in range(num_workers)
        ]
        self.agent = GeminiAgent(api_key=self.api_key)
        
    async def process_query(self, user_query: str, documents: List[str], file_sources: List[str]) -> str:
        """Process a query by distributing work among workers and synthesizing results"""
        start_time = time.time()
        logger.info(f"Orchestrator processing query: {user_query}")
        
        # 1. Analyze query to understand information needs
        query_analysis = await self._analyze_query(user_query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # 2. Create tasks from documents
        tasks = self._create_tasks(user_query, documents, file_sources, query_analysis)
        logger.info(f"Created {len(tasks)} tasks for query processing")
        
        # 3. Execute tasks concurrently using workers
        completed_tasks = await self._execute_tasks(tasks)
        
        # 4. Filter and rank results
        ranked_results = self._rank_results(completed_tasks, query_analysis)
        logger.info(f"Ranked {len(ranked_results)} results from worker tasks")
        
        # 5. Synthesize final response
        final_response = await self._synthesize_response(user_query, ranked_results, query_analysis)
        
        # 6. If no relevant information was found, generate a response using Gemini's general knowledge
        try:
            if not ranked_results or (final_response and "không tìm thấy thông tin" in final_response.lower()):
                logger.info(f"No relevant information found in documents, using Gemini's general knowledge")
                general_response = await self._generate_general_response(user_query)
                if general_response and len(general_response) > 20:
                    final_response = general_response
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            # Continue with the original response if error occurs
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        
        return final_response
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand information needs"""
        # Execute in thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            analysis = await asyncio.get_event_loop().run_in_executor(
                executor, self.agent._analyze_query, query
            )
        
        return analysis or {
            "chủ_đề": "không xác định",
            "từ_khóa": "",
            "file_ưu_tiên": ""
        }
    
    def _create_tasks(self, query: str, documents: List[str], file_sources: List[str], 
                       query_analysis: Dict[str, Any]) -> List[Task]:
        """Create tasks from documents based on query analysis"""
        tasks = []
        
        # Get prioritized file order based on query analysis
        file_priorities = self.agent._determine_file_priority(query_analysis)
        
        # Create a mapping of files to documents for easier lookup
        file_docs_map = {}
        for i, (doc, source) in enumerate(zip(documents, file_sources)):
            if not source:  # Skip if source is None or empty
                continue
            if source not in file_docs_map:
                file_docs_map[source] = []
            file_docs_map[source].append(doc)
        
        # Create tasks for each source file in priority order
        task_id = 0
        for file_name in file_priorities:
            if file_name in file_docs_map:
                # Combine all documents from this file
                combined_context = "\n\n---\n\n".join(file_docs_map[file_name])
                
                # Process the context for better handling
                processed_context = prepare_vietnamese_context(combined_context)
                
                # Limit context size for effective processing
                if len(processed_context) > 28000:
                    processed_context = processed_context[:28000] + "\n\n...(truncated for length)..."
                
                # Create a task for this file
                task = Task(
                    task_id=f"task-{task_id}", 
                    query=query, 
                    context=processed_context,
                    source_file=file_name
                )
                tasks.append(task)
                task_id += 1
        
        # Add tasks for any remaining files not in priorities
        for file_name, docs in file_docs_map.items():
            if file_name not in file_priorities:
                combined_context = "\n\n---\n\n".join(docs)
                processed_context = prepare_vietnamese_context(combined_context)
                
                if len(processed_context) > 28000:
                    processed_context = processed_context[:28000] + "\n\n...(truncated for length)..."
                
                task = Task(
                    task_id=f"task-{task_id}", 
                    query=query, 
                    context=processed_context,
                    source_file=file_name
                )
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    async def _execute_tasks(self, tasks: List[Task]) -> List[Task]:
        """Execute tasks concurrently using available workers"""
        # Use round-robin worker assignment
        num_workers = len(self.workers)
        tasks_per_worker = {worker.worker_id: [] for worker in self.workers}
        
        for i, task in enumerate(tasks):
            worker_idx = i % num_workers
            tasks_per_worker[self.workers[worker_idx].worker_id].append(task)
        
        # Create and gather all worker tasks
        worker_jobs = []
        for worker in self.workers:
            worker_tasks = tasks_per_worker[worker.worker_id]
            if worker_tasks:
                # Process each task for this worker
                for task in worker_tasks:
                    worker_jobs.append(worker.process_task(task))
        
        # Run all tasks concurrently
        if worker_jobs:
            completed_tasks = await asyncio.gather(*worker_jobs)
            return completed_tasks
        return []
    
    def _rank_results(self, completed_tasks: List[Task], query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank and filter results based on relevance to query"""
        results = []
        
        for task in completed_tasks:
            if task.status == "completed" and task.result:
                # Skip results with no relevant information
                content = task.result.get("content", "") 
                if "Không tìm thấy thông tin liên quan" in content:
                    continue
                    
                # Add valid results to the list
                results.append({
                    "content": content,
                    "source_file": task.result.get("source_file"),
                    "relevance_score": task.result.get("relevance_score", 0.0)
                })
        
        # Sort results by relevance score (highest first)
        ranked_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return ranked_results
    
    async def _synthesize_response(self, query: str, ranked_results: List[Dict[str, Any]], 
                                 query_analysis: Dict[str, Any]) -> str:
        """Synthesize a final response from ranked results"""
        if not ranked_results:
            return f"Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong các tài liệu hiện có."
        
        # Extract information from top results (limit to most relevant)
        result_texts = []
        sources = []
        
        # Get the top 3-5 results depending on how many are available
        top_results = ranked_results[:min(5, len(ranked_results))]
        
        for result in top_results:
            # Add content from each result
            result_texts.append(result["content"])
            
            # Track source files for citation
            file_name = result["source_file"]
            # Clean up UUID prefixes if present
            if '_' in file_name and any(c for c in file_name if c == '-'):
                file_name = file_name.split('_', 1)[1] if '_' in file_name else file_name
                
            if file_name not in sources:
                sources.append(file_name)
        
        # Combine result text with proper spacing
        combined_info = "\n\n---\n\n".join(result_texts)
        
        # Format source list for citation
        source_citation = ", ".join(sources)
        
        # Create synthesis prompt
        topic = query_analysis.get("chủ_đề", "không xác định") if query_analysis else "không xác định"
        
        synthesis_prompt = f"""
        Bạn là trợ lý tổng hợp thông tin tuyển sinh của trường Đại học Mở Thành phố Hồ Chí Minh.
        Hãy tổng hợp thông tin từ các kết quả tìm kiếm dưới đây để trả lời câu hỏi một cách đầy đủ và chính xác.
        
        CÂU HỎI: "{query}"
        CHỦ ĐỀ CHÍNH: {topic}
        
        THÔNG TIN TỪ CÁC NGUỒN TÀI LIỆU:
        {combined_info}
        
        HƯỚNG DẪN TỔNG HỢP:
        1. Tổng hợp ngắn gọn, trực tiếp vào nội dung chính mà không cần phần giới thiệu hay mở đầu.
        2. Loại bỏ các câu mang tính nhắc nhở như “Lưu ý”, “Bạn nên truy cập...
        3. Sắp xếp thông tin theo thứ tự logic và liên quan
        4. Đảm bảo phản ánh chính xác mọi dữ liệu, ngày tháng và con số
        5. Chỉ sử dụng thông tin có trong các nguồn, không thêm thông tin ngoài
        6. Xử lý mâu thuẫn giữa các nguồn bằng cách ưu tiên nguồn mới nhất (2025 > 2024)
        7. Tránh lặp lại nội dung giữa các nguồn
        
        YÊU CẦU ĐỊNH DẠNG:
        1. Bắt đầu với tiêu đề <h4 class="text-gradient"> rõ ràng, tóm tắt nội dung chính
        2. Số liệu quan trọng cần làm nổi bật với <strong class="text-accent"> (cho điểm chuẩn, học phí) hoặc <b>
        3. Dữ liệu điểm chuẩn, danh sách ngành, học phí luôn dùng <div class="data-table"> để bao bọc và hiển thị như sau:
           - Điểm chuẩn: <div class="data-table admissions-data">
               <div class="data-row"><div class="data-label">Năm 2024:</div><div class="data-value">22.50</div></div>
               <div class="data-row"><div class="data-label">Năm 2023:</div><div class="data-value">24.00</div></div>
           </div>
        4. Dùng <ul class="feature-list"> và <li> cho các danh sách
        5. Phân đoạn nội dung bằng <div class="content-section mb-3">
        
        Trả lời câu hỏi dựa trên thông tin đã tổng hợp.
        """
        
        # Execute in thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": synthesis_prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048
                }
            }
            
            # Use the agent's API call methods directly
            response_future = asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.agent._call_api(payload)
            )
            
            response_json = await response_future
            final_response = self.agent._process_response(response_json)
        
        # If synthesis failed, create a basic response from the top result
        if not final_response:
            if ranked_results:
                return f"{ranked_results[0]['content']}\n\n<small><i>Thông tin được tìm thấy trong: {source_citation}</i></small>"
            else:
                return f"Xin lỗi, tôi không thể tổng hợp thông tin để trả lời câu hỏi của bạn."
        
        return final_response
        
    async def _generate_general_response(self, query: str) -> str:
        """Generate a response using Gemini's general knowledge when no relevant information is found in documents"""
        # Create a prompt for general knowledge response
        general_prompt = f"""
        Tôi đã tìm kiếm trong các tài liệu của trường nhưng không tìm thấy thông tin liên quan đến câu hỏi này. 
        Vì vậy tôi sẽ trả lời dựa trên kiến thức chung của mình.
        
        CÂU HỎI: "{query}"
        
        Hãy trả lời câu hỏi trên một cách chuyên nghiệp và hữu ích, bắt đầu bằng việc giải thích rằng tôi không tìm thấy thông tin trong tài liệu, 
        và sau đó cung cấp thông tin chung nhất mà tôi biết.
        
        Trả lời nên có định dạng tương tự các câu trả lời khác (có HTML tags).
        """
        
        # Execute in thread pool to avoid blocking
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": general_prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 1024
                    }
                }
                
                # Use the agent's API call methods directly
                response_future = asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self.agent._call_api(payload)
                )
                
                response_json = await response_future
                response_text = self.agent._process_response(response_json) or ""
            
                # Add a disclaimer to make it clear this is not from the documents
                if not response_text and len(response_text) > 10 or not isinstance(response_text, str):
                    if not "<h4>" in response_text[:100].lower():
                        response_text = f"<h4>Thông tin chung</h4>\n{response_text}"
                    if not "không tìm thấy" in response_text.lower():
                        response_text = f"<p><i>Xin lỗi, tôi không tìm thấy thông tin cụ thể về câu hỏi này trong các tài liệu của trường.</i></p>\n{response_text}"
                    if not "<small>" in response_text[-200:].lower():
                        response_text += "\n\n<small><i>Lưu ý: Câu trả lời này dựa trên kiến thức chung, không phải từ tài liệu chính thức của trường.</i></small>"
                
                return response_text
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return ""

async def orchestrate_response(user_query: str, documents: List[str], file_sources: List[str]) -> str:
    """Main entry point for orchestrating document search and response generation"""
    try:
        # Create the orchestrator
        orchestrator = Orchestrator()
        
        # Check if query is likely about programming, technical topic, or others unrelated to admissions
        query_words = user_query.lower().split()
        technical_terms = ['python', 'code', 'function', 'programming', 'javascript', 'hàm', 'code', 'lập trình', 'web', 'algorithm', 'thuật toán']
        
        # Check if query contains technical keywords
        is_technical_query = any(term in query_words for term in technical_terms)
        
        # If query is technical or clearly not about university admissions, use general knowledge directly
        if is_technical_query:
            logger.info(f"Technical query detected, using Gemini's general knowledge: {user_query}")
            general_response = await orchestrator._generate_general_response(user_query)
            return general_response
        
        # Otherwise use the standard orchestrator process
        response = await orchestrator.process_query(user_query, documents, file_sources)
        
        return response
    except Exception as e:
        logger.error(f"Error in orchestration: {e}")
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu: {str(e)}"
