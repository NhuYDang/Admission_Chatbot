import numpy as np
import logging
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TransformerVectorStore:
    """
    A vector store using Sentence Transformers for document similarity
    Provides more semantic understanding compared to TF-IDF
    """
    
    def __init__(self, model_name="bkai-foundation-models/vietnamese-bi-encoder"):
        self.documents = []
        self.file_sources = {}  # Document index -> file source mapping
        self.file_indices = {}  # File source -> list of document indices
        self.file_categories = {}  # Categorization of files by type
        self.model = SentenceTransformer(model_name)
        self.vectors = None
        
    def clear(self):
        """
        Clear all documents and vectors from the store
        """
        self.documents = []
        self.file_sources = {}
        self.file_indices = {}
        self.file_categories = {}
        self.vectors = None
        logger.info("Transformer vector store cleared")
        
    def add_documents(self, documents, file_source=None):
        """
        Add documents to the vector store
        
        Args:
            documents (list): List of text documents to add
            file_source (str): Source file name for these documents
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return
            
        # Save current index before adding new documents
        start_idx = len(self.documents)
        
        # Extract file source from tagged documents if not provided
        if not file_source and documents[0].startswith("SOURCE_FILE:"):
            # Extract file source from the first document's tag
            try:
                source_line = documents[0].split('\n')[0]
                file_source = source_line.replace("SOURCE_FILE:", "").strip()
                
                # Categorize file based on its name
                file_lower = file_source.lower()
                if 'diem' in file_lower or 'chuan' in file_lower:
                    category = 'diem_chuan'
                elif 'hoc_phi' in file_lower or 'bong' in file_lower:
                    category = 'hoc_phi'
                elif 'nganh' in file_lower or 'khoa' in file_lower:
                    category = 'nganh_hoc'
                elif 'co_so' in file_lower or 'vat_chat' in file_lower:
                    category = 'co_so'
                elif 'tuyen_sinh' in file_lower or 'tuyen' in file_lower:
                    category = 'tuyen_sinh'
                else:
                    category = 'other'
                    
                self.file_categories[file_source] = category
                logger.debug(f"Categorized {file_source} as '{category}'")
            except Exception as e:
                logger.error(f"Error extracting source from tagged document: {e}")
        
        # Add new documents
        self.documents.extend(documents)
        
        # Track document indices for each file source
        if file_source:
            end_idx = len(self.documents)
            # Store mapping from indices to file source
            for idx in range(start_idx, end_idx):
                self.file_sources[idx] = file_source
            
            # Store mapping from file source to indices
            if file_source not in self.file_indices:
                self.file_indices[file_source] = []
            self.file_indices[file_source].extend(list(range(start_idx, end_idx)))
            
            logger.debug(f"Tracked {end_idx - start_idx} documents from source '{file_source}'")
        
        # Compute embeddings if we have documents
        if self.documents:
            try:
                # Create embeddings using the sentence transformer model
                with torch.no_grad():
                    self.vectors = self.model.encode(self.documents, convert_to_numpy=True)
                logger.info(f"Added {len(documents)} documents to transformer vector store. Total: {len(self.documents)}")
            except Exception as e:
                logger.error(f"Error creating embeddings with transformer model: {e}")
        else:
            logger.warning("No documents in vector store after attempted addition")
        
    def similarity_search(self, query, k=5, threshold=0.05):
        """
        Find the most similar documents to the query using semantic similarity
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            threshold (float): Minimum similarity score to include a document
            
        Returns:
            list: Top k most similar documents
        """
        if not self.documents or self.vectors is None:
            logger.warning("No documents in transformer vector store")
            return ["No knowledge base available. Please upload PDF files."]
        
        try:
            # Process the query to match the document format
            processed_query = query.lower().strip()
            
            # 1. DETERMINE QUERY TYPE AND PRIORITIZE FILES
            query_file_priorities = self._determine_file_priorities(processed_query)
            
            # 2. SEQUENTIAL SEARCH THROUGH PRIORITIZED FILES
            # First search in prioritized files
            all_results = []
            all_indices = []
            all_scores = []
            
            if query_file_priorities:
                # Try each file category in priority order
                for file_category in query_file_priorities:
                    file_results, file_indices, file_scores = self._search_in_file_category(query, file_category, k, threshold)
                    
                    # Add unique results from this file category
                    for doc, idx, score in zip(file_results, file_indices, file_scores):
                        if idx not in all_indices:
                            all_results.append(doc)
                            all_indices.append(idx)
                            all_scores.append(score)
                    
                    # If we found good matches, we can stop
                    if len(all_results) >= k and max(file_scores) >= threshold * 2:
                        logger.info(f"Found {len(all_results)} good matches in prioritized file category '{file_category}'. Stopping search.")
                        break
            
            # 3. IF NEEDED, EXTEND SEARCH TO ALL FILES
            # If we still don't have enough good results, search all remaining documents
            if len(all_results) < k:
                # Get all remaining documents
                remaining_indices = [i for i in range(len(self.documents)) if i not in all_indices]
                
                if remaining_indices:
                    # Encode the query using the transformer model
                    with torch.no_grad():
                        query_vector = self.model.encode([query], convert_to_numpy=True)
                    
                    # Get vectors for remaining documents
                    remaining_vectors = self.vectors[remaining_indices]
                    
                    # Calculate similarities
                    similarities = cosine_similarity(query_vector, remaining_vectors).flatten()
                    
                    # Sort by similarity
                    sorted_idx = np.argsort(similarities)[::-1]
                    
                    # Add until we have k results or reach the threshold
                    for i in sorted_idx:
                        if similarities[i] >= threshold:
                            doc_idx = remaining_indices[i]
                            if doc_idx not in all_indices:
                                all_indices.append(doc_idx)
                                all_results.append(self.documents[doc_idx])
                                all_scores.append(similarities[i])
                                
                                if len(all_results) >= k:
                                    break
            
            # 4. RETURN RESULTS OR FALLBACK
            if not all_results:
                logger.info(f"No documents found with similarity above threshold {threshold} for query: {query}")
                return ["Tôi không tìm thấy thông tin cụ thể về câu hỏi của bạn trong cơ sở dữ liệu của tôi."]
            
            # Log the file sources for all results
            for i, (doc, idx, score) in enumerate(zip(all_results, all_indices, all_scores)):
                file_source = self.file_sources.get(idx, "Unknown")
                logger.debug(f"Result {i+1} from '{file_source}' (score: {score:.4f}): {doc[:200]}...")
            
            return all_results
        
        except Exception as e:
            logger.error(f"Error in transformer similarity search: {e}")
            return ["Xin lỗi, đã xảy ra lỗi khi tìm kiếm thông tin liên quan."]
    
    def _determine_file_priorities(self, query):
        """
        Determine which file categories should be prioritized based on query content
        """
        query_lower = query.lower()
        priorities = []
        
        # Check for keywords related to each category
        if any(term in query_lower for term in ["điểm", "điểm chuẩn", "điểm xét tuyển", "điểm trúng tuyển"]):
            priorities.append("diem_chuan")
            
        if any(term in query_lower for term in ["học phí", "học bổng", "miễn giảm", "chi phí", "tiền học"]):
            priorities.append("hoc_phi")
            
        if any(term in query_lower for term in ["ngành", "chuyên ngành", "khoa", "chương trình", "đào tạo"]):
            priorities.append("nganh_hoc")
            
        if any(term in query_lower for term in ["cơ sở", "cơ sở vật chất", "địa điểm", "khuôn viên", "phòng học", "khu"]):
            priorities.append("co_so")
            
        if any(term in query_lower for term in ["tuyển sinh", "xét tuyển", "tuyển", "chỉ tiêu", "phương thức", "điều kiện"]):
            priorities.append("tuyen_sinh")
            
        # If no priorities detected, include all categories in reasonable order
        if not priorities:
            priorities = ["tuyen_sinh", "nganh_hoc", "diem_chuan", "hoc_phi", "co_so", "other"]
        else:
            # Add remaining categories at lower priority
            all_categories = ["tuyen_sinh", "nganh_hoc", "diem_chuan", "hoc_phi", "co_so", "other"]
            for category in all_categories:
                if category not in priorities:
                    priorities.append(category)
        
        logger.info(f"Query type analysis for '{query}' determined priorities: {priorities}")
        return priorities
    
    def _search_in_file_category(self, query, category, k, threshold):
        """
        Search for documents within a specific file category using transformer embeddings
        """
        # Find all document indices for this category
        category_indices = []
        for file_source, file_category in self.file_categories.items():
            if file_category == category and file_source in self.file_indices:
                category_indices.extend(self.file_indices[file_source])
        
        if not category_indices:
            return [], [], []
        
        try:
            # Encode the query
            with torch.no_grad():
                query_vector = self.model.encode([query], convert_to_numpy=True)
            
            # Get vectors for documents in this category
            category_vectors = self.vectors[category_indices]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, category_vectors).flatten()
            
            # Sort by similarity
            sorted_idx = np.argsort(similarities)[::-1]
            
            # Collect results that meet the threshold
            results = []
            result_indices = []
            result_scores = []
            
            for i in sorted_idx:
                if similarities[i] >= threshold and len(results) < k:
                    doc_idx = category_indices[i]
                    results.append(self.documents[doc_idx])
                    result_indices.append(doc_idx)
                    result_scores.append(similarities[i])
            
            return results, result_indices, result_scores
            
        except Exception as e:
            logger.error(f"Error searching in file category '{category}': {e}")
            return [], [], [] 