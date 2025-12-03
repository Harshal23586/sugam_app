import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_document_analysis_module(analyzer):
    st.header("📋 AI-Powered Document Sufficiency Analysis")
    
    st.info("Analyze document completeness and generate sufficiency reports for approval processes")
    
    # Generate enhanced dummy document data with realistic patterns
    generate_enhanced_dummy_document_data(analyzer)
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
    selected_institution = st.selectbox(
        "Select Institution",
        current_institutions,
        key="doc_analysis_institution"
    )
    
    approval_type = st.selectbox(
        "Select Approval Type",
        ["new_approval", "renewal_approval", "expansion_approval"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="doc_analysis_approval_type"
    )
    
    # Get institution performance data
    institution_performance = get_institution_performance(selected_institution, analyzer)
    
    # Display performance context
    display_performance_context(institution_performance, selected_institution)
    
    # Display document checklist with enhanced status
    display_enhanced_document_checklist(selected_institution, approval_type, analyzer, institution_performance)
    
    # Analysis and recommendations
    if st.button("🤖 Analyze Document Sufficiency", type="primary"):
        perform_enhanced_document_analysis(selected_institution, approval_type, analyzer, institution_performance)

    pass

def display_enhanced_document_checklist(institution_id, approval_type, analyzer, performance):
    """Display enhanced document checklist with upload dates and status"""
    
    # Get requirements
    requirements = get_document_requirements_by_parameters(approval_type)
    
    # Get uploaded documents for this institution with dates
    uploaded_docs_data = []
    try:
        uploaded_docs_df = analyzer.get_institution_documents(institution_id)
        if not uploaded_docs_df.empty:
            for _, row in uploaded_docs_df.iterrows():
                uploaded_docs_data.append({
                    'name': row['document_name'],
                    'type': row['document_type'],
                    'status': row['status'],
                    'upload_date': row['upload_date']
                })
    except Exception as e:
        st.warning(f"Could not load uploaded documents: {e}")
    
    # Display enhanced document statistics
    st.subheader("📊 Enhanced Document Analysis")
    
    # Calculate statistics
    total_docs = len(uploaded_docs_data)
    uploaded_count = len([d for d in uploaded_docs_data if d['status'] == 'Uploaded'])
    pending_count = len([d for d in uploaded_docs_data if d['status'] == 'Pending'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Uploaded", uploaded_count, delta=f"+{uploaded_count}")
    
    with col3:
        st.metric("Pending", pending_count, delta=f"-{pending_count}", delta_color="inverse")
    
    with col4:
        completion_rate = (uploaded_count / total_docs * 100) if total_docs > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Display mandatory documents with enhanced status
    st.subheader("📋 Mandatory Documents Status")
    
    total_mandatory = 0
    uploaded_mandatory = 0
    pending_mandatory = 0
    
    for parameter, documents in requirements["mandatory"].items():
        with st.expander(f"🔴 {parameter} - Mandatory Documents", expanded=True):
            for doc_template in documents:
                total_mandatory += 1
                
                # Find matching uploaded document
                matching_doc = None
                for uploaded_doc in uploaded_docs_data:
                    if doc_template.lower() in uploaded_doc['name'].lower():
                        matching_doc = uploaded_doc
                        break
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        days_ago = (datetime.now() - pd.to_datetime(matching_doc['upload_date'])).days
                        st.success(f"✅ {doc_template}")
                        st.caption(f"📅 Uploaded {days_ago} days ago")
                    else:
                        st.error(f"❌ {doc_template}")
                        st.caption("⏳ Status: Pending - Institution has failed to submit")
                
                with col2:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        st.markdown("**✅ Uploaded**")
                        uploaded_mandatory += 1
                    else:
                        st.markdown("**🔴 Pending**")
                        pending_mandatory += 1
                
                with col3:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        upload_date = pd.to_datetime(matching_doc['upload_date']).strftime("%d %b %Y")
                        st.markdown(f"**{upload_date}**")
                    else:
                        st.markdown("**OVERDUE**")
    
    # Display supporting documents with enhanced status
    st.subheader("📝 Supporting Documents Status")
    
    total_supporting = 0
    uploaded_supporting = 0
    pending_supporting = 0
    
    for parameter, documents in requirements["supporting"].items():
        with st.expander(f"🟡 {parameter} - Supporting Documents"):
            for doc_template in documents:
                total_supporting += 1
                
                # Find matching uploaded document
                matching_doc = None
                for uploaded_doc in uploaded_docs_data:
                    if doc_template.lower() in uploaded_doc['name'].lower():
                        matching_doc = uploaded_doc
                        break
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        days_ago = (datetime.now() - pd.to_datetime(matching_doc['upload_date'])).days
                        st.info(f"✅ {doc_template}")
                        st.caption(f"📅 Uploaded {days_ago} days ago")
                    else:
                        st.warning(f"⭕ {doc_template}")
                        st.caption("💡 Recommended for enhanced assessment")
                
                with col2:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        st.markdown("**✅ Uploaded**")
                        uploaded_supporting += 1
                    else:
                        st.markdown("**🟡 Optional**")
                        pending_supporting += 1
                
                with col3:
                    if matching_doc and matching_doc['status'] == 'Uploaded':
                        upload_date = pd.to_datetime(matching_doc['upload_date']).strftime("%d %b %Y")
                        st.markdown(f"**{upload_date}**")
                    else:
                        st.markdown("**NOT UPLOADED**")
    
    # Store enhanced counts for analysis
    st.session_state.enhanced_document_counts = {
        'total_mandatory': total_mandatory,
        'uploaded_mandatory': uploaded_mandatory,
        'pending_mandatory': pending_mandatory,
        'total_supporting': total_supporting,
        'uploaded_supporting': uploaded_supporting,
        'pending_supporting': pending_supporting,
        'uploaded_docs_data': uploaded_docs_data,
        'performance': performance
    }
    pass

# ... (move other related functions here)

