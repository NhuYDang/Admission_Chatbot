import PyPDF2
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        text = clean_text(text)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return "No readable text found in the document."
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def clean_text(text):
    """
    Làm sạch văn bản được trích xuất bằng cách loại bỏ khoảng trắng thừa trong khi giữ nguyên các ký tự Unicode
    Các cải tiến cụ thể cho định dạng văn bản tiếng Việt
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Ghi log ăn bản gốc
    logger.debug(f"Original text sample (first 100 chars): {text[:100]}")
    
    # Giữ nguyên các ký tự xuống dòng để duy trì cấu trúc đoạn văn trước khi thực hiện các bước làm sạch khác
    text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
    
    # Thay thế nhiều khoảng trắng bằng một khoảng trắng duy nhất
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Loại bỏ các ký tự không in được trong khi giữ nguyên các ký tự Unicode/tiếng Việt
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Xử lý patterns đặc biệt cho việc tách số trong PDF tiếng Việt (ví dụ: "24.0" có thể xuất hiện dưới dạng "24. 0")
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    # Sửa các khoảng năm có thể bị tách rời
    text = re.sub(r'(\d{4})\s*-\s*(\d{4})', r'\1-\2', text)
    
    # Sửa các khoảng năm với dấu gạch ngang tiếng Việt
    text = re.sub(r'(\d{4})\s*–\s*(\d{4})', r'\1-\2', text)
    
    # Các mẫu từ tiếng Việt phổ biến
    common_prefixes = ['Trường', 'Sinh', 'Học', 'Đại', 'Thí', 'Tuyển', 'Ngành', 'Chương', 'Khoa', 
                    'Phòng', 'Giáo', 'Đào', 'Tạo', 'Viện', 'Bằng', 'Cấp', 'Năm', 'Điểm', 'Chuẩn',
                    'Quản', 'Trị', 'Kinh', 'Doanh', 'Phương', 'Thức', 'Xét', 'Tuyển']
    
    # Chèn khoảng trắng trước các từ viết hoa ở giữa văn bản
    for prefix in common_prefixes:
        pattern = f'([a-z\\s])({prefix})'
        text = re.sub(pattern, r'\1 \2', text)
    
    # Thêm khoảng trắng sau dấu câu nếu chưa có
    text = re.sub(r'([.,;:!?()])([^\s])', r'\1 \2', text)
    
    # Sửa các mẫu cụ thể cho "Năm" theo sau là một năm
    text = re.sub(r'N\s*[aă]\s*m\s*(\d{4})', r'Năm \1', text, flags=re.IGNORECASE)
    
    # Sửa mẫu cho các năm điểm chuẩn (ví dụ: "Năm 2024: 20.75")
    text = re.sub(r'([Nn][ăa]m\s*\d{4})\s*:\s*(\d+[.,]\d+)', r'\1: \2', text)
    
    # Thay thế nhiều ký tự xuống dòng liên tiếp bằng chỉ hai ký tự (để phân tách đoạn)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Sửa khoảng trắng xung quanh các ký tự xuống dòng
    text = re.sub(r'\s*\n\s*', '\n', text)  
    
    # Ghi log văn bản đã làm sạch
    logger.debug(f"Cleaned text sample (first 100 chars): {text[:100]}")
    
    return text.strip()

def chunk_text(text, chunk_size=5000, overlap=500, file_source=None):
    """
    Split text into chunks for processing, optimized for Vietnamese text
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        file_source (str): Source file name to tag in chunks
        
    Returns:
        list: List of text chunks with source tags
    """
    if not text:
        return []
        
    # Log original text length for debugging
    logger.debug(f"Original text length: {len(text)} characters")
    
    # Prepare source tag if file_source is provided
    source_tag = ""
    source_tag_size = 0
    if file_source:
        source_tag = f"SOURCE_FILE:{file_source}\n"
        source_tag_size = len(source_tag)
        # Reduce chunk size to accommodate tag
        effective_chunk_size = chunk_size - source_tag_size
    else:
        effective_chunk_size = chunk_size
    
    # First, try to split by section headers (capitalized with numbers)
    sections = re.split(r'([0-9]+\.[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]+)', text)
    
    # Recombine the sections with their headers
    all_sections = []
    for i in range(0, len(sections)-1, 2):
        if i+1 < len(sections):
            all_sections.append(sections[i] + sections[i+1])
        else:
            all_sections.append(sections[i])
    
    if len(all_sections) <= 1:  # If no sections found, use paragraphs
        # Split by paragraphs (double newlines)
        all_sections = text.split('\n\n')
    
    # Now create chunks from these sections
    chunks = []
    current_chunk = ""
    last_sections = []  # Keep track of recent sections for overlap
    
    for section in all_sections:
        # Clean the section of extra whitespace
        section = section.strip()
        if not section:
            continue
            
        if len(current_chunk) + len(section) + 2 <= effective_chunk_size:
            # Add section to current chunk
            current_chunk += section + "\n\n"
            last_sections.append(section)
            # Limit the stored sections for overlap
            if len(last_sections) > 5:
                last_sections.pop(0)
        else:
            # Chunk is full
            if current_chunk:
                # Add source tag at the beginning of the chunk
                if file_source:
                    tagged_chunk = source_tag + current_chunk.strip()
                else:
                    tagged_chunk = current_chunk.strip()
                chunks.append(tagged_chunk)
            
            # Start new chunk with overlap from previous sections
            overlap_text = "\n\n".join(last_sections[-2:] if len(last_sections) >= 2 else last_sections)
            
            # If overlap text is too large, truncate it
            if len(overlap_text) > overlap:
                # Try to truncate at a sentence boundary
                sentences = re.split(r'([.!?]\s)', overlap_text)
                overlap_text = ""
                for i in range(len(sentences)-1, -1, -2):
                    if i > 0:  # Make sure we have both sentence and delimiter
                        potential_text = sentences[i-1] + sentences[i] + overlap_text
                        if len(potential_text) <= overlap:
                            overlap_text = potential_text
                        else:
                            break
            
            current_chunk = ""
            if overlap_text:
                current_chunk = overlap_text + "\n\n"
            current_chunk += section + "\n\n"
            
            # Reset the sections tracker with current section
            last_sections = [section]
    
    # Add the last chunk if not empty
    if current_chunk:
        # Add source tag at the beginning of the chunk
        if file_source:
            tagged_chunk = source_tag + current_chunk.strip()
        else:
            tagged_chunk = current_chunk.strip()
        chunks.append(tagged_chunk)
    
    # Log chunking results for debugging
    logger.debug(f"Split text into {len(chunks)} chunks")
    if chunks:
        logger.debug(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks)} characters")
    
    return chunks
