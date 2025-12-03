import re
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Tuple
from models.rag_system import RAGDocument, SimpleTextSplitter, SimpleVectorStore
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class RAGDataExtractor:
    def __init__(self):
        try:
            import torch
            device = torch.device('cpu')
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            
            self.embedding_model = SentenceTransformer(
                model_name, 
                device='cpu',
                cache_folder='./model_cache'
            )
            
            if hasattr(self.embedding_model, 'to'):
                self.embedding_model = self.embedding_model.to(device)
            
            test_embedding = self.embedding_model.encode(["test sentence"])
            if test_embedding is not None:
                st.success("✅ RAG System with embeddings initialized successfully")
            else:
                raise Exception("Test embedding failed")
                
            self.text_splitter = SimpleTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.vector_store = None
            self.documents = []
            
        except Exception as e:
            st.warning(f"⚠️ RAG system using lightweight mode: {e}")
            self.embedding_model = None
            self.text_splitter = SimpleTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.vector_store = None
            self.documents = []
            self._setup_lightweight_analyzer()

    def _setup_lightweight_analyzer(self):
        """Setup lightweight text analysis without heavy embeddings"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_matrix = None
            self.document_texts = []
        except:
            self.vectorizer = None

    def build_vector_store(self, documents: List[RAGDocument]):
        """Build vector store from documents with fallback options"""
        if not documents:
            return None
        
        texts = [doc.page_content for doc in documents]
        if not texts:
            return None
            
        try:
            if self.embedding_model is not None:
                embeddings = self.embedding_model.encode(texts)
                text_embeddings = list(zip(texts, embeddings))
                self.vector_store = SimpleVectorStore(self.embedding_model).from_embeddings(text_embeddings)
                self.documents = documents
                st.success(f"✅ Vector store built with {len(documents)} documents using embeddings")
            elif hasattr(self, 'vectorizer') and self.vectorizer is not None:
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.document_texts = texts
                self.documents = documents
                st.success(f"✅ Vector store built with {len(documents)} documents using TF-IDF")
            else:
                self.document_texts = texts
                self.documents = documents
                st.success(f"✅ Documents stored for basic search ({len(documents)} documents)")
                
        except Exception as e:
            st.warning(f"Vector store creation using basic storage: {e}")
            self.document_texts = texts
            self.documents = documents
    
    def extract_comprehensive_data(self, uploaded_files: List) -> Dict[str, Any]:
        """Extract comprehensive data from all uploaded files"""
        all_text = ""
        all_structured_data = {
            'academic_metrics': {},
            'research_metrics': {},
            'infrastructure_metrics': {},
            'governance_metrics': {},
            'student_metrics': {},
            'financial_metrics': {},
            'raw_text': "",
            'file_names': []
        }
    
        documents = []
        processed_count = 0
    
        for file in uploaded_files:
            try:
                text = self.extract_text_from_file(file)
                if not text or not text.strip():
                    st.warning(f"No extractable text found in {file.name}")
                    all_structured_data['file_names'].append(file.name)
                    continue
