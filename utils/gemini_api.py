import os
import json
import re
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def prepare_vietnamese_context(context):
    """
    Prepares Vietnamese text context for better processing by Gemini API
    
    Args:
        context (str): Raw context text with potential formatting issues
        
    Returns:
        str: Processed context optimized for Gemini API
    """
    if not context:
        return ""
        
    # Replace common text issues in Vietnamese PDFs
    replacements = {
        # Common fixes for words that might be concatenated in PDFs
        'THÔNGTINTUYỂNSINH': 'THÔNG TIN TUYỂN SINH',
        'ĐẠIHỌCCHÍNHQUY': 'ĐẠI HỌC CHÍNH QUY',
        'TrườngĐại': 'Trường Đại',
        'họcMở': 'học Mở',
        'ThànhphốHồ': 'Thành phố Hồ',
        'ChíMinh': 'Chí Minh',
        'dựkiến': 'dự kiến',
        'phươnghướng': 'phương hướng',
        'tuyểnsinh': 'tuyển sinh',
        'đạihọc': 'đại học',
        'chínhquy': 'chính quy',
        'năm2025': 'năm 2025',
        'cácnội': 'các nội',
        'dungchính': 'dung chính',
        'nhưsau': 'như sau',
        'Chỉtiêu': 'Chỉ tiêu',
        'ngànhđào': 'ngành đào',
        'tạochuẩn': 'tạo chuẩn',
        'phươngthức': 'phương thức',
        'tốtnghiệp': 'tốt nghiệp',
        'xéttrúng': 'xét trúng',
        'họcbạ': 'học bạ',
        'THPT': 'THPT ',
        'BGD': 'BGD ',
        'ĐT': 'ĐT ',
        'CăncứThông': 'Căn cứ Thông',
        'CăncứĐề': 'Căn cứ Đề',
        'tínchỉ': 'tín chỉ',
        'ngàytháng': 'ngày tháng',
        'kếtquả': 'kết quả',
        'thờiđiểm': 'thời điểm',
        'giáodục': 'giáo dục',
        'ĐàoTạo': 'Đào Tạo',
        'KếHoạch': 'Kế Hoạch',
        'VănBằng': 'Văn Bằng',
        'cógiá': 'có giá',
        'trịtới': 'trị tới',
        'mônxét': 'môn xét',
        'tuyển': 'tuyển',
        '5500': '5500 ',
        '34': '34 ',
        '17': '17 ',
    }
    
    for old, new in replacements.items():
        context = context.replace(old, new)
    
    # Better formatting with section headers
    # Identify section headers (numbers followed by dot and capital letters)
    context = re.sub(r'(\d+\.)([A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])', r'\n\n\1 \2', context)
    
    # Make sure there's a period at the end of each sentence
    context = re.sub(r'([^.!?\s])\s+([A-Z])', r'\1. \2', context)
    
    # Better paragraph and section formatting
    context = re.sub(r'([.?!])\s+', r'\1\n', context)  # New line after each sentence
    context = re.sub(r'\n{3,}', '\n\n', context)  # Replace excessive newlines
    
    # Structure important section headers for clarity
    important_sections = ['THÔNG TIN TUYỂN SINH', 'Chỉ tiêu', 'Phương thức', 'Điều kiện', 'Hồ sơ', 'Thời gian']
    
    for section in important_sections:
        context = context.replace(section, f'\n\n### {section.upper()} ###\n')
    
    # Final cleanup
    context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
    context = re.sub(r'\n\s+', '\n', context)  # Clean up spaces at line beginnings
    context = re.sub(r'\n{3,}', '\n\n', context)  # Limit consecutive newlines to 2
    
    return context

class GeminiAgent:
    """
    Agent-based interface for the Gemini API with ability to retrieve information,
    make decisions, and execute actions to achieve specific goals.
    """
    
    def __init__(self, api_key=None):
        """Initialize the agent with an API key"""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.memory = []
        self.file_priorities = {}
        self.action_history = []
        self.last_actions = {}
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("API key not configured. Please set the GEMINI_API_KEY environment variable.")
    
    def _call_api(self, payload):
        """Make a call to the Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Add retry logic for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise an exception for 4XX and 5XX responses
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def _process_response(self, response_json):
        """Extract the text from the Gemini API response"""
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        
        logger.error(f"Unexpected API response structure: {response_json}")
        return None
    
    def _record_action(self, action_type, details, result=None):
        """Record an action taken by the agent for auditing and debugging"""
        timestamp = datetime.now().isoformat()
        action = {
            "timestamp": timestamp,
            "type": action_type,
            "details": details,
            "result": result
        }
        
        self.action_history.append(action)
        self.last_actions[action_type] = action
        return action
    
    def _analyze_query(self, user_query):
        """Analyze the user query to determine the information needed"""
        # System prompt for query analysis
        analysis_prompt = f"""
        Bạn là một trợ lý phân tích câu hỏi. Hãy phân tích câu hỏi sau về TUYỂN SINH và xác định:
        
        1. CHỦ ĐỀ CHÍNH (điểm chuẩn/học phí/ngành học/cơ sở vật chất/tuyển sinh/khác)
        2. CÁC TỪ KHÓA QUAN TRỌNG
        3. LOẠI FILE PDF phù hợp nhất để tìm kiếm thông tin (diem_chuan.pdf/hoc_phi_hoc_bong.pdf/thong_tin_nganh_hoc.pdf/co_so_vat_chat.pdf/thong_tin_tuyen_sinh_2025.pdf/thong_tin_tuyen_sinh_2024.pdf)
        
        Hãy trả lời cấu trúc JSON với các trường: chủ_đề, từ_khóa, file_ưu_tiên.
        
        Câu hỏi: "{user_query}"
        """
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": analysis_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 1024
            }
        }
        
        try:
            response_json = self._call_api(payload)
            analysis_text = self._process_response(response_json)
            
            if not analysis_text:
                return None
                
            # Try to extract JSON from the response
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, analysis_text)
            
            if match:
                json_str = match.group(0)
                try:
                    analysis_data = json.loads(json_str)
                    self._record_action("query_analysis", {"query": user_query}, analysis_data)
                    return analysis_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from analysis response: {json_str}")
                    
            # If JSON extraction fails, use regex to extract key information
            topic_match = re.search(r'chủ[\s_]đề[^:]*:\s*([^\n,\.]+)', analysis_text, re.IGNORECASE)
            keywords_match = re.search(r'từ[\s_]khóa[^:]*:\s*([^\n\.]+)', analysis_text, re.IGNORECASE)
            files_match = re.search(r'file[\s_]ưu[\s_]tiên[^:]*:\s*([^\n\.]+)', analysis_text, re.IGNORECASE)
            
            result = {
                "chủ_đề": topic_match.group(1).strip() if topic_match else "unknown",
                "từ_khóa": keywords_match.group(1).strip() if keywords_match else "",
                "file_ưu_tiên": files_match.group(1).strip() if files_match else ""
            }
            
            self._record_action("query_analysis_fallback", {"query": user_query}, result)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return None
    
    def _determine_file_priority(self, query_analysis):
        """Determine the priority of files to search based on query analysis"""
        # Default priority if analysis fails
        default_priority = [
            "thong_tin_tuyen_sinh_2025.pdf",
            "thong_tin_tuyen_sinh_2024.pdf",
            "diem_chuan.pdf",
            "hoc_phi_hoc_bong.pdf",
            "thong_tin_nganh_hoc.pdf",
            "co_so_vat_chat.pdf"
        ]
        
        if not query_analysis:
            return default_priority
        
        # Extract main topic from analysis
        topic = query_analysis.get("chủ_đề", "").lower()
        file_preference = query_analysis.get("file_ưu_tiên", "").lower()
        
        # Determine priority based on topic
        if "điểm" in topic or "điểm chuẩn" in topic:
            priority = ["diem_chuan.pdf", "thong_tin_tuyen_sinh_2025.pdf", "thong_tin_tuyen_sinh_2024.pdf"]
        elif "học phí" in topic or "phí" in topic or "học bổng" in topic:
            priority = ["hoc_phi_hoc_bong.pdf", "thong_tin_tuyen_sinh_2025.pdf"]
        elif "ngành" in topic or "khoa" in topic or "chuyên ngành" in topic or "đào tạo" in topic:
            priority = ["thong_tin_nganh_hoc.pdf", "thong_tin_tuyen_sinh_2025.pdf", "thong_tin_tuyen_sinh_2024.pdf"]
        elif "vị trí" in topic or "việc làm" in topic or "cơ hội" in topic or "ngành nghề" in topic or "nghề nghiệp" in topic:
            priority = ["thong_tin_nganh_hoc.pdf"]
        elif "cơ sở" in topic or "vật chất" in topic or "thư viện" in topic:
            priority = ["co_so_vat_chat.pdf", "thong_tin_tuyen_sinh_2025.pdf"]
        elif "tuyển sinh" in topic or "tuyển" in topic or "chỉ tiêu" or "tổ hợp" in topic or "trường" in topic:
            priority = ["thong_tin_tuyen_sinh_2025.pdf", "thong_tin_tuyen_sinh_2024.pdf","OU_info.pdf"]
        else:
            # For unclassified topics, use the analysis file recommendation if available
            if file_preference:
                priority = [f for f in default_priority if f in file_preference]
                # Add remaining files that weren't mentioned
                priority.extend([f for f in default_priority if f not in priority])
            else:
                priority = default_priority
        
        # Record the determined priority
        self._record_action("file_prioritization", {"topic": topic}, priority)
        return priority

    def search_and_extract(self, user_query, context_docs, file_sources=None):
        """Search through the context documents and extract relevant information"""
        # First analyze the query to understand what we're looking for
        query_analysis = self._analyze_query(user_query)
        
        # Determine file priority based on the analysis
        file_priority = self._determine_file_priority(query_analysis)
        
        # Process context by categorizing documents by their source files
        file_contexts = {}
        if file_sources and context_docs:
            # Group context documents by their source files
            for i, doc in enumerate(context_docs):
                if i < len(file_sources) and file_sources[i]:
                    file_name = file_sources[i]
                    # Handle UUIDs in filenames - extract the actual filename after the underscore
                    if '_' in file_name and any(c for c in file_name if c == '-'):
                        file_name = file_name.split('_', 1)[1] if '_' in file_name else file_name
                    
                    if file_name not in file_contexts:
                        file_contexts[file_name] = []
                    file_contexts[file_name].append(doc)
        
        # Prioritize the documents based on the file priority
        prioritized_docs = []
        prioritized_sources = []
        
        # Add documents from prioritized files first
        for file_name in file_priority:
            if file_name in file_contexts:
                prioritized_docs.extend(file_contexts[file_name])
                prioritized_sources.extend([file_name] * len(file_contexts[file_name]))
        
        # Add any remaining documents
        for file_name, docs in file_contexts.items():
            if file_name not in file_priority:
                prioritized_docs.extend(docs)
                prioritized_sources.extend([file_name] * len(docs))
        
        # If no files were prioritized, use the original order
        if not prioritized_docs and context_docs:
            prioritized_docs = context_docs
            prioritized_sources = file_sources if file_sources else ["unknown"] * len(context_docs)
        
        # Record the search action
        self._record_action("document_search", {
            "query": user_query,
            "file_priority": file_priority,
            "document_count": len(prioritized_docs),
            "sources": prioritized_sources
        })
        
        return prioritized_docs, prioritized_sources, query_analysis
    
    def formulate_task_plan(self, user_query, prioritized_docs, prioritized_sources, query_analysis):
        """Formulate a plan to answer the user's query based on available information"""
        # Prepare a summary of available documents
        doc_summary = ""
        if prioritized_docs and prioritized_sources:
            doc_summary = "\n\nTài liệu có sẵn:\n"
            files_seen = set()
            for i, (doc, source) in enumerate(zip(prioritized_docs, prioritized_sources)):
                if source not in files_seen and len(files_seen) < 5:
                    files_seen.add(source)
                    doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                    doc_summary += f"- {source}: {doc_preview}\n"
            if len(prioritized_sources) > 5:
                doc_summary += f"- Và {len(prioritized_sources) - 5} tài liệu khác.\n"
        
        # Extract query information
        topic = query_analysis.get("chủ_đề", "không xác định") if query_analysis else "không xác định"
        keywords = query_analysis.get("từ_khóa", "") if query_analysis else ""
        
        # Create planning prompt
        planning_prompt = f"""
        Bạn là một trợ lý lập kế hoạch tìm kiếm thông tin. Với câu hỏi sau của người dùng, hãy lập ra kế hoạch để tìm kiếm thông tin chính xác và đầy đủ.
        
        Câu hỏi: "{user_query}"
        
        Phân tích ban đầu:
        - Chủ đề: {topic}
        - Từ khóa: {keywords}
        {doc_summary}
        
        Hãy lập kế hoạch tìm kiếm thông tin với các bước cụ thể, bao gồm:
        1. MỤC TIÊU CHÍNH: Xác định rõ mục tiêu cần đạt được khi trả lời câu hỏi
        2. NGUỒN THÔNG TIN ƯU TIÊN: Xác định nguồn thông tin cần tra cứu trước tiên
        3. THÔNG TIN CẦN TÌM: Liệt kê các thông tin cụ thể cần tìm kiếm
        4. CÁC BƯỚC THỰC HIỆN: Các bước cần thực hiện để trả lời câu hỏi
        5. CÁCH KIỂM TRA: Làm thế nào để đảm bảo thông tin tìm được là chính xác
        
        Hãy trả lời dưới dạng JSON với các trường: mục_tiêu, nguồn_ưu_tiên, thông_tin_cần_tìm, các_bước_thực_hiện, cách_kiểm_tra.
        """
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": planning_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        try:
            response_json = self._call_api(payload)
            plan_text = self._process_response(response_json)
            
            if not plan_text:
                return None
                
            # Try to extract JSON from the response
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, plan_text)
            
            if match:
                json_str = match.group(0)
                try:
                    plan_data = json.loads(json_str)
                    self._record_action("task_planning", {"query": user_query}, plan_data)
                    return plan_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from plan response: {json_str}")
            
            # If JSON extraction fails, extract key sections using regex
            goal_match = re.search(r'mục[\s_]tiêu[^:]*:\s*([^\n]+)', plan_text, re.IGNORECASE)
            sources_match = re.search(r'nguồn[\s_]ưu[\s_]tiên[^:]*:\s*([^\n]+)', plan_text, re.IGNORECASE)
            info_match = re.search(r'thông[\s_]tin[\s_]cần[\s_]tìm[^:]*:\s*([^\n\d]+)', plan_text, re.IGNORECASE)
            steps_match = re.search(r'các[\s_]bước[\s_]thực[\s_]hiện[^:]*:\s*([^\n\d]+)', plan_text, re.IGNORECASE)
            
            # Create a basic plan structure
            result = {
                "mục_tiêu": goal_match.group(1).strip() if goal_match else "Tìm thông tin chính xác để trả lời câu hỏi",
                "nguồn_ưu_tiên": sources_match.group(1).strip() if sources_match else "Các file PDF liên quan",
                "thông_tin_cần_tìm": info_match.group(1).strip() if info_match else "Thông tin liên quan đến câu hỏi",
                "các_bước_thực_hiện": steps_match.group(1).strip() if steps_match else "Tìm kiếm thông tin trong các tài liệu"
            }
            
            self._record_action("task_planning_fallback", {"query": user_query}, result)
            return result
            
        except Exception as e:
            logger.error(f"Error creating task plan: {e}")
            return None
    
    def execute_plan(self, task_plan, user_query, prioritized_docs, prioritized_sources):
        """Execute the task plan to answer the user's query"""
        # Combine documents with their sources
        doc_with_sources = []
        for i, (doc, source) in enumerate(zip(prioritized_docs, prioritized_sources)):
            if len(doc.strip()) > 0:  # Only add non-empty documents
                # Extract the simple filename if it has a UUID prefix
                simple_source = source
                if '_' in source and any(c for c in source if c == '-'):
                    simple_source = source.split('_', 1)[1] if '_' in source else source
                
                doc_with_sources.append(f"FILE: {simple_source}\n{doc}")
        
        # Create a summary of the plan being executed
        plan_summary = "\n\nKẾ HOẠCH TÌM KIẾM THÔNG TIN:\n"
        if task_plan:
            plan_summary += f"- Mục tiêu: {task_plan.get('mục_tiêu', '')}\n"
            plan_summary += f"- Thông tin cần tìm: {task_plan.get('thông_tin_cần_tìm', '')}\n"
            plan_summary += f"- Nguồn ưu tiên: {task_plan.get('nguồn_ưu_tiên', '')}\n"
        
        # Prepare context with relevant documents
        context = "\n\n---\n\n".join(doc_with_sources)
        
        # Process and limit context size
        processed_context = prepare_vietnamese_context(context)
        if len(processed_context) > 28000:
            processed_context = processed_context[:28000] + "\n\n...(truncated for length)..."
        
        # Create the execution prompt
        execution_prompt = f"""
        Bạn là trợ lý tư vấn tuyển sinh thông minh của trường Đại học Mở Thành phố Hồ Chí Minh. Bạn đang thực hiện nhiệm vụ trả lời câu hỏi sau đây:
        
        Câu hỏi: "{user_query}"
        {plan_summary}
        
        
        {processed_context}
        
        Thực hiện các bước sau:
        1. Đọc kỹ các tài liệu
        2. Tìm các thông tin liên quan đến câu hỏi
        3. Tổng hợp thông tin từ các nguồn
        4. Đảm bảo câu trả lời đầy đủ và chính xác
        
        YÊU CẦU VỀ ĐỊNH DẠNG CÂU TRẢ LỜI:
        1. Bắt đầu với tiêu đề <h4> rõ ràng, tóm tắt nội dung chính.
        2. Số liệu quan trọng cần làm nổi bật với <strong> hoặc <b>.
        3. Dùng <table>, <tr>, <td> cho dữ liệu có cấu trúc (bảng điểm, học phí các ngành...).
        4. Dùng <ul> và <li> cho các danh sách.
        
        Hãy trả lời câu hỏi của người dùng dựa trên thông tin trong tài liệu:
        """
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": execution_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }
        
        try:
            response_json = self._call_api(payload)
            answer = self._process_response(response_json)
            
            if not answer:
                return "Xin lỗi, tôi không thể tìm thấy thông tin để trả lời câu hỏi của bạn."
            
            # Record the execution action
            self._record_action("plan_execution", {
                "query": user_query,
                "documents_used": len(prioritized_docs),
                "sources_used": list(set(prioritized_sources))
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return f"Xin lỗi, đã xảy ra lỗi khi tìm kiếm thông tin: {str(e)}"
    
    def reflect_and_improve(self, user_query, answer, prioritized_sources):
        """Reflect on the answer and suggest improvements"""
        # Create a list of unique sources
        unique_sources = list(set(prioritized_sources))
        sources_list = ", ".join(unique_sources[:5])
        if len(unique_sources) > 5:
            sources_list += f" và {len(unique_sources) - 5} file khác"
        
        reflection_prompt = f"""
        Hãy đánh giá câu trả lời tư vấn tuyển sinh sau đây và xác định cách cải thiện nó.
        
        Câu hỏi: "{user_query}"
        
        Câu trả lời: "{answer[:2000]}"
        
        Nguồn thông tin sử dụng: {sources_list}
        
        Hãy đánh giá câu trả lời theo các tiêu chí sau:
        1. Độ chính xác: Thông tin có chính xác không?
        2. Độ đầy đủ: Câu trả lời có đầy đủ thông tin để trả lời câu hỏi không?
        3. Định dạng: Có đúng định dạng yêu cầu không (tiêu đề, nhấn mạnh, bảng, danh sách...)?
        
        Trả lời dưới dạng JSON với các trường: mức_độ_hoàn_thành, điểm_mạnh, điểm_yếu, đề_xuất_cải_thiện.
        """
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": reflection_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        try:
            response_json = self._call_api(payload)
            reflection_text = self._process_response(response_json)
            
            if not reflection_text:
                return None
                
            # Try to extract JSON from the response
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, reflection_text)
            
            if match:
                json_str = match.group(0)
                try:
                    reflection_data = json.loads(json_str)
                    self._record_action("answer_reflection", {"query": user_query}, reflection_data)
                    return reflection_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from reflection: {json_str}")
                    
            return None
            
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return None
    
    def improve_answer(self, user_query, initial_answer, reflection, prioritized_docs, prioritized_sources):
        """Improve the answer based on reflection"""
        if not reflection or not any(prioritized_docs):
            return initial_answer
        
        # Extract improvement suggestions from reflection
        weaknesses = reflection.get("điểm_yếu", "")
        suggestions = reflection.get("đề_xuất_cải_thiện", "")
        completion_level = reflection.get("mức_độ_hoàn_thành", "")
        
        # Only proceed with improvement if the completion level is less than 90%
        # or if there are significant weaknesses or suggestions
        if ("90%" in completion_level or "100%" in completion_level) and not weaknesses and not suggestions:
            return initial_answer
        
        # Combine documents with their sources for context
        doc_with_sources = []
        for i, (doc, source) in enumerate(zip(prioritized_docs[:3], prioritized_sources[:3])):
            if len(doc.strip()) > 0:  # Only add non-empty documents
                # Extract the simple filename if it has a UUID prefix
                simple_source = source
                if '_' in source and any(c for c in source if c == '-'):
                    simple_source = source.split('_', 1)[1] if '_' in source else source
                
                doc_with_sources.append(f"FILE: {simple_source}\n{doc}")
        
        # Prepare limited context with relevant documents to fit within token limits
        context = "\n\n---\n\n".join(doc_with_sources)
        processed_context = prepare_vietnamese_context(context)
        if len(processed_context) > 10000:  # Use a smaller limit for improvement context
            processed_context = processed_context[:10000] + "\n\n...(truncated for length)..."
        
        # Create the improvement prompt
        improvement_prompt = f"""
        Bạn là trợ lý tư vấn tuyển sinh thông minh của trường Đại học Mở Thành phố Hồ Chí Minh.
        
        Câu hỏi: "{user_query}"
        
        Câu trả lời ban đầu:
        "{initial_answer[:1500]}"
        
        Đánh giá và đề xuất cải thiện:
        - Điểm yếu: {weaknesses}
        - Đề xuất cải thiện: {suggestions}
        
        Dưới đây là một số thông tin thêm từ tài liệu để giúp cải thiện câu trả lời:
        {processed_context}
        
        Hãy chỉnh sửa và cải thiện câu trả lời ban đầu, giữ nguyên định dạng HTML, đảm bảo đáp ứng các yêu cầu sau:
        1. Sửa mọi thông tin không chính xác
        2. Bổ sung thông tin còn thiếu
        3. Đảm bảo định dạng đúng (tiêu đề <h4>, nhấn mạnh <strong>, bảng <table>, danh sách <ul><li>)
        
        Chỉ trả về phiên bản cải tiến của câu trả lời, không thêm giải thích.
        """
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": improvement_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }
        
        try:
            response_json = self._call_api(payload)
            improved_answer = self._process_response(response_json)
            
            if not improved_answer or len(improved_answer) < 100:
                return initial_answer
            
            # Record the improvement action
            self._record_action("answer_improvement", {
                "query": user_query,
                "initial_length": len(initial_answer),
                "improved_length": len(improved_answer),
                "completion_level": completion_level
            })
            
            return improved_answer
            
        except Exception as e:
            logger.error(f"Error improving answer: {e}")
            return initial_answer

def generate_response(user_query, context, chat_history=None):
    """
    Generate a response using Google's Gemini API with agent-based approach
    
    Args:
        user_query (str): User's question
        context (str): Context retrieved from RAG system
        chat_history (list): Previous messages for context
        
    Returns:
        str: Generated response
    """  
    try:
        # Initialize the Gemini Agent
        agent = GeminiAgent()
        
        # Extract file sources from context documents
        file_sources = []
        processed_docs = []
        
        if isinstance(context, list):
            context_docs = context
        else:
            # If context is a string, split it into a list of documents
            context_docs = [context]
        
        for doc in context_docs:
            # Parse source file information from the document
            try:
                if isinstance(doc, str) and doc.startswith("SOURCE_FILE:"):
                    # Extract the source file name from the tag
                    source_line = doc.split('\n')[0]  # Get the first line with source info
                    file_id = source_line.replace("SOURCE_FILE:", "").strip()
                    
                    # Add to list of sources
                    file_sources.append(file_id)
                    
                    # Remove the source tag line for the content
                    processed_content = "\n".join(doc.split('\n')[1:])
                    processed_docs.append(processed_content)
                else:
                    # No source tag, use as is
                    processed_docs.append(doc)
                    file_sources.append(None)  # No source for this document
            except Exception as e:
                logger.error(f"Error parsing document source: {e}")
                processed_docs.append(doc)  # Use original if any error
                file_sources.append(None)  # No source for this document
        
        # Step 1: Search and extract information from documents
        prioritized_docs, prioritized_sources, query_analysis = agent.search_and_extract(
            user_query, processed_docs, file_sources
        )
        
        # Step 2: Formulate a task plan
        task_plan = agent.formulate_task_plan(
            user_query, prioritized_docs, prioritized_sources, query_analysis
        )
        
        # Step 3: Execute the plan to generate an initial answer
        initial_answer = agent.execute_plan(
            task_plan, user_query, prioritized_docs, prioritized_sources
        )
        
        # Step 4: Reflect on the answer and identify improvements
        reflection = agent.reflect_and_improve(
            user_query, initial_answer, prioritized_sources
        )
        
        # Step 5: Improve the answer based on reflection
        final_answer = agent.improve_answer(
            user_query, initial_answer, reflection, prioritized_docs, prioritized_sources
        )
        
        return final_answer
    
    except Exception as e:
        logger.error(f"Error in agent-based response generation: {e}")
        
        # Fall back to regular response generation if agent approach fails
        context_text = "\n\n".join(processed_docs) if 'processed_docs' in locals() else context
        
        # Process Vietnamese text for better results
        processed_context = prepare_vietnamese_context(context_text)
        
        # Limit context size
        if len(processed_context) > 28000:
            processed_context = processed_context[:28000] + "\n\n...(truncated for length)..."
        
        # Create system prompt
        system_prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh thông minh cho trường Đại học Mở Thành phố Hồ Chí Minh. 
        Dựa trên thông tin từ các tài liệu, hãy trả lời câu hỏi sau đây chính xác và đầy đủ. 
        Có thể tóm gọn lại thông tin miễn trả lời đúng trọng tâm câu hỏi.
        
        Thông tin từ tài liệu:
        {processed_context}
        
        Câu hỏi: {user_query}
        """
        
        # Get API key
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "Error: API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        # Make API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": system_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                response_json = response.json()
                
                if "candidates" in response_json and len(response_json["candidates"]) > 0:
                    candidate = response_json["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                
                return "Xin lỗi, tôi không thể tìm được thông tin phù hợp."
            else:
                return f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu. Lỗi: {response.status_code}"
                
        except Exception as inner_e:
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu: {str(inner_e)}"
