import streamlit as st

def create_institution_document_upload(analyzer, user):
    st.subheader("📤 Document Upload Portal")
    
    st.info("Upload required documents for approval processes")
    
    approval_type = st.selectbox(
        "Select Approval Type",
        ["new_approval", "renewal_approval", "expansion_approval"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="inst_approval_type"
    )
    
    uploaded_files = st.file_uploader(
        "Upload Institutional Documents",
        type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png'],
        accept_multiple_files=True,
        help="Upload all required documents for your application"
    )
    
    if uploaded_files:
        st.subheader("📝 Document Type Assignment")
        document_types = []
        for i, file in enumerate(uploaded_files):
            doc_type = st.selectbox(
                f"Document type for: {file.name}",
                [
                    "affidavit_legal_status", "land_documents", "building_plan_approval", 
                    "financial_solvency_certificate", "faculty_recruitment_plan", 
                    "academic_curriculum", "annual_reports", "research_publications",
                    "placement_records", "other"
                ],
                key=f"inst_doc_type_{i}"
            )
            document_types.append(doc_type)
        
        if st.button("💾 Upload Documents"):
            analyzer.save_uploaded_documents(user['institution_id'], uploaded_files, document_types)
            st.success("✅ Documents uploaded successfully!")
