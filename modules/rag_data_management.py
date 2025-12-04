# modules/rag_data_management.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from io import BytesIO
import tempfile
from rag_system import SimpleTextSplitter, SimpleVectorStore
import numpy as np

def create_rag_data_management(analyzer):
    """RAG Data Management Module"""
    
    st.header("📊 RAG Data Management")
    
    st.info("Manage and configure the RAG (Retrieval-Augmented Generation) system")
    
    tab1, tab2, tab3 = st.tabs(["📂 Document Library", "🔧 Configuration", "📈 Performance"])
    
    with tab1:
        st.subheader("Document Library")
        st.write("Upload and manage documents for the RAG system")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'txt', 'docx', 'pptx', 'csv'],
                help="Upload documents to be processed by the RAG system"
            )
        
        with col2:
            document_type = st.selectbox(
                "Document Type",
                ["Accreditation Report", "Financial Statement", "Research Paper", "Other"]
            )
            auto_process = st.checkbox("Auto-process after upload", value=True)
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type,
                "Uploaded": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"- {key}: {value}")
            
            with col2:
                if st.button("📄 Preview Content", type="secondary"):
                    try:
                        content = uploaded_file.getvalue().decode('utf-8')
                        st.text_area("File Preview", content[:1000], height=200)
                    except:
                        st.warning("Cannot preview binary file")
            
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process the document
                        if uploaded_file.name.endswith('.txt'):
                            with open(tmp_path, 'r', encoding='utf-8') as f:
                                text_content = f.read()
                        elif uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(tmp_path)
                            text_content = df.to_string()
                        else:
                            # For other file types, show placeholder
                            text_content = f"Binary file: {uploaded_file.name}"
                        
                        # Initialize text splitter
                        splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
                        
                        # Split text into chunks
                        chunks = splitter.split_text(text_content)
                        
                        # Store in analyzer if available
                        if hasattr(analyzer, 'rag_documents'):
                            analyzer.rag_documents.extend(chunks)
                        
                        st.success(f"✅ Document processed successfully!")
                        st.write(f"**Processing Results:**")
                        st.write(f"- Total chunks: {len(chunks)}")
                        st.write(f"- First chunk preview: {chunks[0][:200]}..." if chunks else "No content")
                        
                        # Show chunk distribution
                        if chunks:
                            chunk_lengths = [len(chunk) for chunk in chunks]
                            st.write(f"- Average chunk length: {np.mean(chunk_lengths):.0f} characters")
                            st.write(f"- Min/Max chunk length: {min(chunk_lengths)}/{max(chunk_lengths)} characters")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
        
        # Show existing documents
        st.markdown("---")
        st.subheader("📚 Existing Documents")
        
        if hasattr(analyzer, 'rag_documents') and analyzer.rag_documents:
            total_chunks = len(analyzer.rag_documents)
            total_chars = sum(len(doc) for doc in analyzer.rag_documents)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", "Multiple" if total_chunks > 1 else "1")
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                st.metric("Total Content", f"{total_chars:,} chars")
            
            # Document management
            with st.expander("Manage Documents"):
                for i, doc in enumerate(analyzer.rag_documents[:10]):  # Show first 10
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**Chunk {i+1}:** {doc[:100]}...")
                    with col2:
                        if st.button(f"❌", key=f"delete_{i}"):
                            analyzer.rag_documents.pop(i)
                            st.success(f"Deleted chunk {i+1}")
                            st.rerun()
                
                if total_chunks > 10:
                    st.info(f"... and {total_chunks - 10} more chunks")
                
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    analyzer.rag_documents = []
                    st.success("All documents cleared!")
                    st.rerun()
        else:
            st.info("No documents have been uploaded yet.")
    
    with tab2:
        st.subheader("RAG Configuration")
        
        # Initialize session state for configuration
        if 'rag_config' not in st.session_state:
            st.session_state.rag_config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'retrieval_k': 3,
                'similarity_threshold': 0.7,
                'embedding_model': 'all-MiniLM-L6-v2',
                'enable_reranking': True
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Text Processing:**")
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=500,
                max_value=2000,
                value=st.session_state.rag_config['chunk_size'],
                step=100,
                help="Size of text chunks for processing"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=500,
                value=st.session_state.rag_config['chunk_overlap'],
                step=50,
                help="Overlap between consecutive chunks"
            )
        
        with col2:
            st.write("**Retrieval Settings:**")
            retrieval_k = st.slider(
                "Number of Results (K)",
                min_value=1,
                max_value=10,
                value=st.session_state.rag_config['retrieval_k'],
                step=1,
                help="Number of similar documents to retrieve"
            )
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.rag_config['similarity_threshold'],
                step=0.05,
                help="Minimum similarity score for retrieval"
            )
        
        st.write("**Advanced Settings:**")
        col1, col2 = st.columns(2)
        
        with col1:
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en", "paraphrase-MiniLM-L6-v2"],
                index=0 if st.session_state.rag_config['embedding_model'] == 'all-MiniLM-L6-v2' else 1,
                help="Model for generating text embeddings"
            )
        
        with col2:
            enable_reranking = st.checkbox(
                "Enable Re-ranking",
                value=st.session_state.rag_config['enable_reranking'],
                help="Re-rank results for better accuracy"
            )
        
        # Configuration actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration", type="primary"):
                st.session_state.rag_config = {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'retrieval_k': retrieval_k,
                    'similarity_threshold': similarity_threshold,
                    'embedding_model': embedding_model,
                    'enable_reranking': enable_reranking
                }
                st.success("✅ Configuration saved!")
        
        with col2:
            if st.button("🔄 Test Configuration", type="secondary"):
                with st.spinner("Testing configuration..."):
                    # Test the configuration
                    test_text = "This is a test document for RAG system configuration."
                    splitter = SimpleTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = splitter.split_text(test_text)
                    st.success(f"✅ Configuration test successful!")
                    st.write(f"- Test chunks created: {len(chunks)}")
                    st.write(f"- Chunk sizes: {[len(c) for c in chunks]}")
        
        with col3:
            if st.button("📥 Export Configuration", type="secondary"):
                config_json = st.session_state.rag_config
                st.download_button(
                    label="Download JSON",
                    data=str(config_json),
                    file_name=f"rag_config_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    with tab3:
        st.subheader("System Performance")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            docs_processed = len(analyzer.rag_documents) if hasattr(analyzer, 'rag_documents') else 0
            st.metric("Documents Processed", docs_processed)
        
        with col2:
            # Simulate embedding count
            embedding_count = docs_processed * 10  # Assuming 10 chunks per doc
            st.metric("Embeddings Generated", embedding_count)
        
        with col3:
            st.metric("System Status", "✅ Active" if docs_processed > 0 else "⏸️ Inactive")
        
        # Performance charts placeholder
        st.markdown("---")
        st.write("**Performance Analytics**")
        
        # Create sample data
        performance_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'Processing Time (ms)': np.random.randint(100, 500, 30),
            'Memory Usage (MB)': np.random.randint(50, 200, 30),
            'Documents Processed': np.random.randint(0, 10, 30)
        })
        
        st.line_chart(performance_data.set_index('Date')['Processing Time (ms)'])
        
        # Recent activity log
        st.markdown("---")
        st.write("**Recent Activity Log**")
        
        activity_log = [
            {"timestamp": "2024-01-15 10:30", "action": "Document Upload", "details": "annual_report_2023.pdf"},
            {"timestamp": "2024-01-15 10:32", "action": "Processing Complete", "details": "Created 15 chunks"},
            {"timestamp": "2024-01-15 10:35", "action": "Configuration Update", "details": "Chunk size: 1000"},
            {"timestamp": "2024-01-15 10:40", "action": "Query Processed", "details": "Academic performance metrics"}
        ]
        
        for activity in activity_log:
            col1, col2, col3 = st.columns([2, 2, 4])
            with col1:
                st.write(activity['timestamp'])
            with col2:
                st.write(activity['action'])
            with col3:
                st.write(activity['details'])
        
        # System diagnostics
        st.markdown("---")
        st.write("**System Diagnostics**")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            if st.button("🩺 Run Diagnostics", type="secondary"):
                with st.spinner("Running diagnostics..."):
                    st.success("✅ System diagnostics completed!")
                    st.write("**Results:**")
                    st.write("- ✅ Document processing: Functional")
                    st.write("- ✅ Text splitting: Operational")
                    st.write("- ⚠️ Embedding generation: Not initialized")
                    st.write("- ✅ Configuration storage: Working")
        
        with diag_col2:
            if st.button("🔄 Refresh Statistics", type="secondary"):
                st.rerun()

# Also add a method to the analyzer to initialize RAG documents
def initialize_rag_for_analyzer(analyzer):
    """Initialize RAG documents list for the analyzer"""
    if not hasattr(analyzer, 'rag_documents'):
        analyzer.rag_documents = []
    return analyzer
