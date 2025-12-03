# institution/documents.py
"""
Institution Document Upload Module

Handles document upload, categorization, and sufficiency analysis
for institutional approval processes.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict
import json

def create_institution_document_upload(analyzer, user):
    """
    Document upload portal for institutions
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: Dictionary containing user information
    """
    st.subheader("📤 Document Upload Portal")
    
    st.info("""
    **Upload required documents for approval processes**
    
    Ensure all documents are:
    - In PDF, DOC, DOCX, or image formats
    - Properly labeled and dated
    - Clear and legible
    - Under 10MB per file
    """)
    
    # Step 1: Select approval type
    col1, col2 = st.columns(2)
    
    with col1:
        approval_type = st.selectbox(
            "Select Approval Type",
            ["new_approval", "renewal_approval", "expansion_approval"],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Type of approval you are applying for",
            key="inst_approval_type"
        )
    
    with col2:
        document_parameter = st.selectbox(
            "Select Document Parameter",
            [
                "Curriculum",
                "Faculty Resources", 
                "Learning and Teaching",
                "Research and Innovation",
                "Extracurricular & Co-curricular Activities",
                "Community Engagement",
                "Green Initiatives",
                "Governance and Administration",
                "Infrastructure Development",
                "Financial Resources and Management"
            ],
            help="Parameter category for the document",
            key="doc_parameter"
        )
    
    # Step 2: Document selection based on parameter
    mandatory_document, supporting_document = get_document_options(document_parameter)
    
    col1, col2 = st.columns(2)
    
    with col1:
        mandatory_doc = st.selectbox(
            "Select Mandatory Document",
            mandatory_document,
            help="Required document for this parameter",
            key="mandatory_doc"
        )
    
    with col2:
        supporting_doc = st.selectbox(
            "Select Supporting Document",
            supporting_document,
            help="Optional but recommended document",
            key="supporting_doc"
        )
    
    # Step 3: File upload
    st.markdown("---")
    st.markdown("### 📎 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload Institutional Documents",
        type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png', 'jpeg'],
        accept_multiple_files=True,
        help="You can upload multiple files at once. Max 10MB per file.",
        key="inst_doc_upload"
    )
    
    if uploaded_files:
        # Display uploaded files
        st.markdown("#### 📋 Uploaded Files Preview")
        
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{i+1}. {file.name}**")
                st.caption(f"Size: {file.size / 1024:.1f} KB | Type: {file.type}")
            
            with col2:
                # Document type assignment
                doc_type = st.selectbox(
                    f"Type for {file.name[:20]}...",
                    ["mandatory", "supporting", "additional"],
                    key=f"doc_type_{i}"
                )
            
            with col3:
                if st.button("🗑️", key=f"remove_{i}", help="Remove this file"):
                    # Remove file logic (in real app, you'd handle this)
                    st.warning("File removal would be implemented here")
        
        # Upload button
        if st.button("💾 Upload All Documents", type="primary", use_container_width=True):
            process_uploaded_documents(
                analyzer, user, uploaded_files, approval_type,
                mandatory_doc, supporting_doc, document_parameter
            )
    
    # Step 4: Document requirements and guidelines
    st.markdown("---")
    st.markdown("### 📋 Document Guidelines")
    
    with st.expander("📄 Document Requirements", expanded=False):
        show_document_requirements(approval_type)
    
    with st.expander("🛠️ Upload Tips", expanded=False):
        show_upload_tips()
    
    with st.expander("📊 Document Status", expanded=False):
        show_document_status(analyzer, user)

def get_document_options(parameter: str):
    """
    Get document options based on selected parameter
    
    Args:
        parameter: Selected document parameter
    
    Returns:
        Tuple of (mandatory_documents, supporting_documents)
    """
    document_maps = {
        "Curriculum": {
            "mandatory": [
                "Curriculum framework and syllabus documents for all programs",
                "Course outlines with learning objectives and outcomes",
                "Evidence of curriculum review and revision processes",
                "Academic calendar and course schedules"
            ],
            "supporting": [
                "Innovative teaching-learning materials developed",
                "Records of curriculum innovation and updates",
                "Industry interface documents for curriculum design"
            ]
        },
        "Faculty Resources": {
            "mandatory": [
                "Faculty recruitment policy and procedures",
                "Faculty qualification records and biodata",
                "Faculty-student ratio documentation"
            ],
            "supporting": [
                "Faculty achievement and award records",
                "Participation in development programs",
                "Faculty research and publication records"
            ]
        },
        # ... Add other parameters similarly
        "Financial Resources and Management": {
            "mandatory": [
                "Annual financial statements and audit reports",
                "Budget allocation and utilization certificates",
                "Salary expenditure records"
            ],
            "supporting": [
                "Financial planning documents",
                "Resource mobilization records",
                "Investment in academic development"
            ]
        }
    }
    
    if parameter in document_maps:
        return document_maps[parameter]["mandatory"], document_maps[parameter]["supporting"]
    
    # Default fallback
    return ["Select document"], ["Select document"]

def process_uploaded_documents(analyzer, user, uploaded_files, approval_type,
                            mandatory_doc, supporting_doc, document_parameter):
    """
    Process uploaded documents
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: User information
        uploaded_files: List of uploaded files
        approval_type: Type of approval
        mandatory_doc: Selected mandatory document
        supporting_doc: Selected supporting document
        document_parameter: Selected document parameter
    """
    if not uploaded_files:
        st.warning("⚠️ No files selected for upload")
        return
    
    with st.spinner(f"Uploading {len(uploaded_files)} document(s)..."):
        try:
            # Save documents to database
            document_types = []
            for i, file in enumerate(uploaded_files):
                doc_type = st.session_state.get(f"doc_type_{i}", "additional")
                document_types.append(doc_type)
            
            analyzer.save_uploaded_documents(
                user['institution_id'],
                uploaded_files,
                document_types
            )
            
            # Show success message
            st.success(f"✅ Successfully uploaded {len(uploaded_files)} document(s)!")
            
            # Analyze document sufficiency
            analyze_document_sufficiency(analyzer, user, approval_type)
            
            # Show next steps
            show_upload_next_steps()
            
        except Exception as e:
            st.error(f"❌ Error uploading documents: {str(e)}")

def analyze_document_sufficiency(analyzer, user, approval_type):
    """
    Analyze document sufficiency for approval type
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: User information
        approval_type: Type of approval
    """
    st.markdown("---")
    st.markdown("### 📊 Document Sufficiency Analysis")
    
    try:
        # Get uploaded documents
        uploaded_docs = analyzer.get_institution_documents(user['institution_id'])
        
        if uploaded_docs.empty:
            st.info("No documents uploaded yet")
            return
        
        # Calculate statistics
        total_docs = len(uploaded_docs)
        mandatory_docs = len(uploaded_docs[uploaded_docs['document_type'] == 'mandatory'])
        supporting_docs = len(uploaded_docs[uploaded_docs['document_type'] == 'supporting'])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", total_docs)
        
        with col2:
            st.metric("Mandatory", mandatory_docs)
        
        with col3:
            st.metric("Supporting", supporting_docs)
        
        with col4:
            if total_docs > 0:
                sufficiency = (mandatory_docs / (mandatory_docs + supporting_docs)) * 100
                st.metric("Sufficiency", f"{sufficiency:.1f}%")
        
        # Recommendations
        if mandatory_docs == 0:
            st.error("⚠️ **Critical**: No mandatory documents uploaded!")
            st.write("Upload mandatory documents to proceed with approval process.")
        elif mandatory_docs < 3:
            st.warning("⚠️ **Attention**: Few mandatory documents uploaded.")
            st.write("Consider uploading more mandatory documents.")
        else:
            st.success("✅ **Good progress**: Adequate documents uploaded.")
            st.write("You can now proceed to data submission.")
        
    except Exception as e:
        st.error(f"Error analyzing documents: {str(e)}")

def show_document_requirements(approval_type):
    """
    Show document requirements for approval type
    
    Args:
        approval_type: Type of approval
    """
    requirements = {
        "new_approval": {
            "mandatory": [
                "Affidavit of legal status",
                "Land and building documents",
                "Financial solvency certificate",
                "Faculty recruitment plan",
                "Academic curriculum documents"
            ],
            "supporting": [
                "Market demand analysis",
                "Five-year development plan",
                "Industry partnership agreements"
            ]
        },
        "renewal_approval": {
            "mandatory": [
                "Previous approval letters",
                "Annual reports (last 3 years)",
                "Financial audit reports",
                "Faculty and student data"
            ],
            "supporting": [
                "NAAC accreditation report",
                "Research publications",
                "Placement records"
            ]
        },
        "expansion_approval": {
            "mandatory": [
                "Current status report",
                "Expansion justification",
                "Additional infrastructure plans",
                "Enhanced faculty plan"
            ],
            "supporting": [
                "Stakeholder feedback",
                "Market analysis",
                "Financial projections"
            ]
        }
    }
    
    if approval_type in requirements:
        req = requirements[approval_type]
        
        st.markdown("#### 📋 Mandatory Documents")
        for doc in req['mandatory']:
            st.write(f"• {doc}")
        
        st.markdown("#### 📝 Supporting Documents")
        for doc in req['supporting']:
            st.write(f"• {doc}")

def show_upload_tips():
    """Show tips for successful document upload"""
    st.markdown("#### 💡 Upload Tips")
    
    tips = [
        "📄 **Use PDF format** for text documents for better compatibility",
        "🏷️ **Name files clearly** (e.g., 'NAAC_Report_2023.pdf')",
        "📏 **Keep file sizes under 10MB** for faster upload",
        "📅 **Include dates** in document names when applicable",
        "🔒 **Ensure sensitive information** is properly redacted",
        "📋 **Group related documents** in single uploads when possible",
        "🔄 **Check document orientation** before uploading scanned documents",
        "🎯 **Focus on mandatory documents** first for faster processing"
    ]
    
    for tip in tips:
        st.write(tip)

def show_document_status(analyzer, user):
    """
    Show current document status for the institution
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: User information
    """
    try:
        docs = analyzer.get_institution_documents(user['institution_id'])
        
        if docs.empty:
            st.info("No documents uploaded yet")
            return
        
        # Create status summary
        status_counts = docs['status'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Uploaded", status_counts.get('Uploaded', 0))
        
        with col2:
            st.metric("Pending", status_counts.get('Pending', 0))
        
        with col3:
            st.metric("Reviewed", status_counts.get('Reviewed', 0))
        
        # Recent uploads
        st.markdown("#### 📅 Recent Uploads")
        recent_docs = docs.sort_values('upload_date', ascending=False).head(5)
        
        for _, doc in recent_docs.iterrows():
            days_ago = (datetime.now() - pd.to_datetime(doc['upload_date'])).days
            st.write(f"• {doc['document_name']} ({days_ago} days ago)")
            
    except Exception as e:
        st.error(f"Error loading document status: {str(e)}")

def show_upload_next_steps():
    """Show next steps after successful upload"""
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")
    
    steps = [
        "📝 **Submit Institutional Data**: Complete the Basic or Systematic Data Form",
        "📊 **Track Progress**: Monitor your submission in 'My Submissions'",
        "🔄 **Check Requirements**: Review document requirements in 'Requirements Guide'",
        "⏳ **Wait for Review**: Documents will be reviewed within 5-7 working days",
        "📧 **Check Email**: You'll receive notifications at each stage"
    ]
    
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")

if __name__ == "__main__":
    # Test the module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from core.analyzer import InstitutionalAIAnalyzer
    
    st.set_page_config(page_title="Document Upload Test", layout="wide")
    
    dummy_user = {
        'institution_id': 'INST_0001',
        'institution_name': 'Test University',
        'contact_person': 'Dr. Test User',
        'email': 'test@university.edu',
        'role': 'Institution'
    }
    
    analyzer = InstitutionalAIAnalyzer()
    create_institution_document_upload(analyzer, dummy_user)
