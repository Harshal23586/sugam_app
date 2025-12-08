# modules/rag_dashboard.py
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Initialize PDF processor variables
pdfplumber = None
PyPDF2 = None
PDF_PROCESSOR = None

# Try to import PDF libraries with fallbacks
try:
    import pdfplumber as pdfplumber_module
    pdfplumber = pdfplumber_module
    PDF_PROCESSOR = "pdfplumber"
    st.sidebar.info("✓ pdfplumber available")
except ImportError:
    st.sidebar.warning("pdfplumber not installed")

try:
    import PyPDF2 as PyPDF2_module
    PyPDF2 = PyPDF2_module
    if PDF_PROCESSOR is None:
        PDF_PROCESSOR = "PyPDF2"
    st.sidebar.info("✓ PyPDF2 available")
except ImportError:
    st.sidebar.warning("PyPDF2 not installed")


def create_rag_dashboard(analyzer):
    """Main RAG Dashboard for document intelligence"""
    
    st.title("🧠 RAG Document Intelligence System")
    
    st.markdown("""
    **Retrieval-Augmented Generation System** for intelligent document analysis, compliance checking, 
    and evidence-based decision making for institutional approvals.
    """)
    
    # Initialize RAG system if not exists
    if not hasattr(analyzer, 'rag_system'):
        from modules.rag_core import InstitutionalRAGSystem
        analyzer.rag_system = InstitutionalRAGSystem(analyzer)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Document Processing",
        "🔍 Document Intelligence",
        "📊 Compliance Analysis",
        "🎯 Performance Insights",
        "⚙️ System Configuration"
    ])
    
    with tab1:
        _document_processing_tab(analyzer)
    
    with tab2:
        _document_intelligence_tab(analyzer)
    
    with tab3:
        _compliance_analysis_tab(analyzer)
    
    with tab4:
        _performance_insights_tab(analyzer)
    
    with tab5:
        _system_configuration_tab(analyzer)

def _document_processing_tab(analyzer):
    """Document upload and processing"""
    
    st.header("📁 Document Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Institutional Document",
            type=['pdf', 'txt', 'docx', 'pptx', 'csv'],
            help="Upload NAAC reports, financial statements, accreditation documents, etc."
        )
    
    with col2:
        institution_id = st.selectbox(
            "Institution",
            options=analyzer.historical_data['institution_id'].unique()[:20],
            help="Select the institution this document belongs to"
        )
        
        document_type = st.selectbox(
            "Document Type",
            ["NAAC Report", "Financial Statement", "Audit Report", "Annual Report", 
             "Research Publications", "Infrastructure Report", "Faculty Details", "Other"]
        )
    
    if uploaded_file and institution_id and document_type:
        # Process the file based on type
        if uploaded_file.name.lower().endswith('.pdf'):
            content = process_pdf_document(uploaded_file, analyzer)
        else:
            # For non-PDF files
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        
        if content:
            # Preview
            with st.expander("📄 Document Preview"):
                st.text_area("Content", content[:2000], height=300)
            
            # Additional metadata for PDFs
            if uploaded_file.name.lower().endswith('.pdf'):
                col1, col2 = st.columns(2)
                with col1:
                    year = st.number_input("Report Year", min_value=2000, max_value=2024, value=2023)
                with col2:
                    st.write(f"File: {uploaded_file.name}")
            
            # Process button
            if st.button("🔍 Process Document", type="primary"):
                with st.spinner("Analyzing document with RAG system..."):
                    metadata = {
                        'institution_id': institution_id,
                        'document_type': document_type,
                        'filename': uploaded_file.name,
                        'upload_date': datetime.now().isoformat(),
                        'size_kb': len(content) / 1024,
                        'year': year if 'year' in locals() else 2023
                    }
                    
                    result = analyzer.rag_system.process_institutional_document(content, metadata)
                    
                    st.success("✅ Document processed successfully!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sections Found", result['total_sections'])
                    with col2:
                        st.metric("Content Chunks", result['total_chunks'])
                    with col3:
                        if 'metrics' in result and 'assessment_score' in result['metrics']:
                            st.metric("Assessment Score", f"{result['metrics']['assessment_score']:.1f}")
                    
                    # Show extracted sections
                    with st.expander("📋 Extracted Sections"):
                        for i, section in enumerate(result['sections']):
                            st.write(f"**{i+1}. {section['title']}** ({section['section_type']})")
                            st.write(f"Chunks: {section['chunk_count']}")
                            if 'compliance' in section:
                                st.progress(section['compliance']['compliance_score'] / 100)
                                st.caption(f"Compliance: {section['compliance']['compliance_score']:.1f}%")
                    
                    # Show extracted metrics
                    if result['metrics']:
                        with st.expander("📈 Extracted Metrics"):
                            for key, value in result['metrics'].items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def _document_intelligence_tab(analyzer):
    """Document query and intelligence"""
    
    st.header("🔍 Document Intelligence")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        institution_id = st.selectbox(
            "Select Institution",
            options=analyzer.historical_data['institution_id'].unique()[:20],
            key="query_institution"
        )
    
    with col2:
        query = st.text_input(
            "Ask a question about this institution's documents",
            placeholder="e.g., 'What is the NAAC grade?' or 'Show me financial performance metrics'"
        )
    
    if institution_id and query:
        if st.button("🔎 Search Documents", type="primary"):
            with st.spinner("Searching across all documents..."):
                results = analyzer.rag_system.query_institution(institution_id, query)
                
                if results['found']:
                    st.success(f"✅ Found {len(results['results'])} relevant sections")
                    
                    # Display summary
                    st.subheader("📋 Summary")
                    st.write(results['summary'])
                    
                    # Display individual results
                    st.subheader("📄 Relevant Document Sections")
                    for i, result in enumerate(results['results']):
                        with st.expander(f"Result {i+1}: {result['document_type']} ({result['year']})"):
                            st.write("**Context:**")
                            st.write(result['context'])
                            st.write("**Metadata:**", result['metadata'])
                else:
                    st.warning("No relevant documents found. Try different keywords.")

def _compliance_analysis_tab(analyzer):
    """Compliance checking and analysis"""
    
    st.header("📊 Compliance Analysis")
    
    institution_id = st.selectbox(
        "Select Institution for Compliance Check",
        options=analyzer.historical_data['institution_id'].unique()[:20],
        key="compliance_institution"
    )
    
    if institution_id and st.button("✅ Run Compliance Analysis", type="primary"):
        with st.spinner("Analyzing compliance across all documents..."):
            analysis = analyzer.rag_system.analyze_institution_performance(institution_id)
            
            if 'error' in analysis:
                st.error(analysis['error'])
            else:
                st.success(f"✅ Analyzed {analysis['total_documents']} documents")
                
                # Compliance summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Types", len(analysis['document_types']))
                with col2:
                    if 'overall_compliance' in analysis:
                        st.metric("Overall Compliance", f"{analysis['overall_compliance']:.1f}%")
                with col3:
                    st.metric("Years Covered", len(analysis['years_covered']))
                
                # Document types breakdown
                st.subheader("📂 Document Inventory")
                doc_types_df = pd.DataFrame({
                    'Document Type': list(analysis['document_types'].keys()),
                    'Count': list(analysis['document_types'].values())
                })
                st.bar_chart(doc_types_df.set_index('Document Type'))
                
                # Compliance details
                st.subheader("📋 Compliance Status by Document Type")
                for doc_type, compliance in analysis['compliance_status'].items():
                    with st.expander(f"{doc_type} ({compliance['compliance_score']:.1f}%)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**✅ Found Elements:**")
                            for elem in compliance['found_elements'][:5]:
                                st.write(f"- {elem}")
                        with col2:
                            if compliance['missing_elements']:
                                st.write("**❌ Missing Elements:**")
                                for elem in compliance['missing_elements']:
                                    st.write(f"- {elem}")
                
                # Risk indicators
                if analysis['risk_indicators']:
                    st.subheader("⚠️ Risk Indicators Found")
                    for risk in analysis['risk_indicators'][:5]:
                        st.warning(f"**{risk['document_type']} ({risk['year']}):** {risk['indicator']}")
                        st.caption(f"Context: {risk['context']}")

def _performance_insights_tab(analyzer):
    """Performance insights and analytics"""
    
    st.header("🎯 Performance Insights")
    
    # Get all institutions with RAG documents
    if hasattr(analyzer.rag_system, 'document_index'):
        institutions_with_docs = set(
            info['institution_id'] 
            for info in analyzer.rag_system.document_index.values()
            if info['institution_id']
        )
    else:
        institutions_with_docs = set()
    
    if not institutions_with_docs:
        st.info("Upload documents in the Document Processing tab to see insights")
        return
    
    selected_institutions = st.multiselect(
        "Select Institutions to Compare",
        options=list(institutions_with_docs),
        default=list(institutions_with_docs)[:3] if institutions_with_docs else []
    )
    
    if selected_institutions:
        # Compare document coverage
        coverage_data = []
        for inst_id in selected_institutions:
            doc_count = sum(
                1 for info in analyzer.rag_system.document_index.values() 
                if info['institution_id'] == inst_id
            )
            coverage_data.append({'Institution': inst_id, 'Documents': doc_count})
        
        coverage_df = pd.DataFrame(coverage_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Document Coverage")
            fig = px.bar(coverage_df, x='Institution', y='Documents')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Analysis Depth")
            # Calculate average chunks per document
            depth_data = []
            for inst_id in selected_institutions:
                total_chunks = 0
                doc_count = 0
                for doc_id, info in analyzer.rag_system.document_index.items():
                    if info['institution_id'] == inst_id:
                        total_chunks += info['total_chunks']
                        doc_count += 1
                
                if doc_count > 0:
                    depth_data.append({
                        'Institution': inst_id,
                        'Avg Chunks per Doc': total_chunks / doc_count
                    })
            
            if depth_data:
                depth_df = pd.DataFrame(depth_data)
                fig = px.bar(depth_df, x='Institution', y='Avg Chunks per Doc')
                st.plotly_chart(fig, use_container_width=True)
        
        # Generate insights
        st.subheader("💡 Key Insights")
        insights = []
        for inst_id in selected_institutions:
            # Analyze this institution
            analysis = analyzer.rag_system.analyze_institution_performance(inst_id)
            
            if 'overall_compliance' in analysis:
                insights.append(f"**{inst_id}**: {analysis['overall_compliance']:.1f}% overall compliance")
            
            if analysis.get('risk_indicators'):
                insights.append(f"**{inst_id}**: {len(analysis['risk_indicators'])} risk indicators found")
        
        for insight in insights:
            st.write(f"- {insight}")

def process_pdf_document(uploaded_file, analyzer):
    """Process uploaded PDF file"""
    
    if pdfplumber is None and PyPDF2 is None:
        st.error("No PDF processing libraries installed. Please add 'pdfplumber' or 'PyPDF2' to requirements.txt")
        return None
    
    # Read the file as bytes
    pdf_bytes = uploaded_file.read()
    text = ""
    
    # Try pdfplumber first if available
    if pdfplumber is not None:
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}")
    
    # Fall back to PyPDF2 if available
    if PyPDF2 is not None and not text.strip():
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e2:
            st.error(f"PyPDF2 failed: {e2}")
    
    if not text.strip():
        st.error("No text could be extracted from the PDF")
        return None
    
    return text
    
def _system_configuration_tab(analyzer):
    """RAG system configuration"""
    
    st.header("⚙️ RAG System Configuration")
    
    if 'rag_config' not in st.session_state:
        st.session_state.rag_config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'similarity_threshold': 0.7,
            'enable_smart_splitting': True,
            'auto_extract_metrics': True,
            'enable_compliance_checking': True
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Processing")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=500,
            max_value=2000,
            value=st.session_state.rag_config['chunk_size'],
            step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=st.session_state.rag_config['chunk_overlap'],
            step=50
        )
        smart_splitting = st.checkbox(
            "Enable Smart Document Splitting",
            value=st.session_state.rag_config['enable_smart_splitting']
        )
    
    with col2:
        st.subheader("🔍 Retrieval Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_config['similarity_threshold'],
            step=0.05
        )
        auto_extract = st.checkbox(
            "Auto-extract Metrics",
            value=st.session_state.rag_config['auto_extract_metrics']
        )
        compliance_checking = st.checkbox(
            "Enable Compliance Checking",
            value=st.session_state.rag_config['enable_compliance_checking']
        )
    
    # System status
    st.subheader("📊 System Status")
    
    if hasattr(analyzer, 'rag_system'):
        total_docs = len(analyzer.rag_system.documents)
        total_chunks = sum(
            info['total_chunks'] 
            for info in analyzer.rag_system.document_index.values()
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", total_docs)
        with col2:
            st.metric("Total Chunks", total_chunks)
        with col3:
            st.metric("Institutions Covered", len(set(
                info['institution_id'] 
                for info in analyzer.rag_system.document_index.values()
                if info['institution_id']
            )))
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.rag_config = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'similarity_threshold': similarity_threshold,
            'enable_smart_splitting': smart_splitting,
            'auto_extract_metrics': auto_extract,
            'enable_compliance_checking': compliance_checking
        }
        st.success("✅ Configuration saved!")
