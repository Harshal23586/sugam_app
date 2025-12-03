import streamlit as st
import pandas as pd

def create_institution_dashboard(analyzer, user):
    if not user:
        st.error("No user data available")
        return
        
    st.header(f"🏛️ Institution Dashboard - {user.get('institution_name', 'Unknown')}")
    
    # Display institution overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Institution ID", user.get('institution_id', 'N/A'))
    with col2:
        st.metric("Contact Person", user.get('contact_person', 'N/A'))
    with col3:
        st.metric("Email", user.get('email', 'N/A'))
    with col4:
        st.metric("Role", user.get('role', 'N/A'))
    
    # Navigation tabs
    institution_tabs = st.tabs([
        "📤 Document Upload", 
        "📝 Basic Data Submission",
        "🏛️ Systematic Data Form",
        "📊 My Submissions",
        "📋 Requirements Guide",
        "🔄 Approval Workflow"
    ])
    
    with institution_tabs[0]:
        from institution.documents import create_institution_document_upload
        create_institution_document_upload(analyzer, user)
    
    with institution_tabs[1]:
        from institution.forms import create_institution_data_submission
        create_institution_data_submission(analyzer, user)
    
    with institution_tabs[2]:
        from institution.forms import create_systematic_data_submission_form
        create_systematic_data_submission_form(analyzer, user)
    
    with institution_tabs[3]:
        from institution.submissions import create_institution_submissions_view
        create_institution_submissions_view(analyzer, user)
    
    with institution_tabs[4]:
        create_institution_requirements_guide(analyzer)
    
    with institution_tabs[5]:
        create_institution_approval_workflow(analyzer, user)

def create_institution_approval_workflow(analyzer, user):
    st.subheader("🔄 Institution Approval Workflow")
    st.info("Track your approval application status and follow the workflow steps")
    
    # Show current submission status
    submissions = analyzer.get_institution_submissions(user['institution_id'])
    
    if len(submissions) > 0:
        st.subheader("📋 Current Application Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = submissions.iloc[0]['status']
            if status == 'Under Review':
                st.warning(f"**Status:** {status}")
            elif status == 'Approved':
                st.success(f"**Status:** {status}")
            elif status == 'Rejected':
                st.error(f"**Status:** {status}")
            else:
                st.info(f"**Status:** {status}")
        
        with col2:
            st.write(f"**Submission Date:** {submissions.iloc[0]['submitted_date']}")
        
        with col3:
            if submissions.iloc[0]['reviewed_by']:
                st.write(f"**Reviewed by:** {submissions.iloc[0]['reviewed_by']}")
    else:
        st.info("No submissions found. Submit your data to start the approval process.")

def create_institution_requirements_guide(analyzer):
    st.subheader("📋 Approval Requirements Guide")
    
    requirements = analyzer.document_requirements
    
    for approval_type, docs in requirements.items():
        with st.expander(f"{approval_type.replace('_', ' ').title()} Requirements"):
            st.write("**Mandatory Documents:**")
            for doc in docs['mandatory']:
                st.write(f"• {doc.replace('_', ' ').title()}")
            
            st.write("**Supporting Documents:**")
            for doc in docs['supporting']:
                st.write(f"• {doc.replace('_', ' ').title()}")
