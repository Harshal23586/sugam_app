# institution/submissions.py
"""
Institution Submissions Module

This module handles viewing and tracking of institution submissions,
requirements guides, and approval workflows.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

def create_institution_submissions_view(analyzer, user):
    """Display institution's submissions and their status"""
    st.subheader("📊 My Submissions & Status")
    
    # Get submissions for the institution
    submissions = analyzer.get_institution_submissions(user['institution_id'])
    
    if len(submissions) > 0:
        st.info(f"📋 **Total Submissions:** {len(submissions)}")
        
        # Create tabs for different submission types
        submission_types = submissions['submission_type'].unique()
        
        for sub_type in submission_types:
            type_submissions = submissions[submissions['submission_type'] == sub_type]
            
            with st.expander(f"{sub_type.replace('_', ' ').title()} ({len(type_submissions)})"):
                for _, submission in type_submissions.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**Submitted:** {submission['submitted_date']}")
                    
                    with col2:
                        status = submission['status']
                        if status == 'Approved':
                            st.success(f"✅ {status}")
                        elif status == 'Rejected':
                            st.error(f"❌ {status}")
                        elif status == 'Under Review':
                            st.info(f"⏳ {status}")
                        else:
                            st.write(f"📝 {status}")
                    
                    with col3:
                        if submission['reviewed_by']:
                            st.write(f"**By:** {submission['reviewed_by']}")
                    
                    # Display review comments if available
                    if submission['review_comments']:
                        st.info(f"**Comments:** {submission['review_comments']}")
                    
                    st.markdown("---")
    else:
        st.info("📭 No submissions found. Use the Data Submission tabs to submit your institutional data.")
        
        # Quick submission options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Submit Basic Data", use_container_width=True):
                st.session_state.active_tab = "📝 Basic Data Submission"
                st.rerun()
        
        with col2:
            if st.button("🏛️ Submit Systematic Data", use_container_width=True):
                st.session_state.active_tab = "🏛️ Systematic Data Form"
                st.rerun()

def create_institution_requirements_guide(analyzer):
    """Display approval requirements guide"""
    st.subheader("📋 Approval Requirements Guide")
    
    st.info("""
    **Guidance for Institutional Approvals**
    
    This section provides detailed information about the documents and data required 
    for different types of institutional approvals.
    """)
    
    # Get requirements from analyzer
    requirements = analyzer.document_requirements
    
    # Create tabs for different approval types
    for approval_type, docs in requirements.items():
        with st.expander(f"📄 {approval_type.replace('_', ' ').title()} Requirements"):
            
            st.markdown("#### 📋 Mandatory Documents")
            mandatory_col1, mandatory_col2 = st.columns([3, 1])
            
            with mandatory_col1:
                for doc in docs['mandatory']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
            
            with mandatory_col2:
                st.warning("**Required**")
            
            st.markdown("---")
            
            st.markdown("#### 📝 Supporting Documents")
            supporting_col1, supporting_col2 = st.columns([3, 1])
            
            with supporting_col1:
                for doc in docs['supporting']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
            
            with supporting_col2:
                st.info("**Recommended**")
            
            # Tips and notes
            st.markdown("---")
            st.markdown("#### 💡 Tips & Notes")
            
            if approval_type == "new_approval":
                st.write("""
                - Submit all documents in PDF format
                - Ensure documents are properly signed and dated
                - Include supporting financial documents for at least 3 years
                - Curriculum documents should be approved by academic council
                """)
            elif approval_type == "renewal_approval":
                st.write("""
                - Include progress reports since last approval
                - Highlight improvements and achievements
                - Address any conditions from previous approval
                - Include feedback from stakeholders
                """)
            elif approval_type == "expansion_approval":
                st.write("""
                - Clearly justify the need for expansion
                - Include detailed financial projections
                - Show evidence of infrastructure readiness
                - Demonstrate faculty recruitment plans
                """)

def create_institution_approval_workflow(analyzer, user):
    """Display approval workflow and status"""
    st.subheader("🔄 Institution Approval Workflow")
    
    st.info("""
    **Track Your Approval Application Status**
    
    Follow the step-by-step workflow to understand where your application stands
    and what actions are required next.
    """)
    
    # Get current submissions
    submissions = analyzer.get_institution_submissions(user['institution_id'])
    
    # Determine current stage
    current_stage = "Not Started"
    
    if len(submissions) > 0:
        latest_submission = submissions.iloc[0]
        current_status = latest_submission['status']
        
        # Map status to stage
        if current_status == 'Under Review':
            current_stage = "Under Committee Review"
        elif current_status == 'Approved':
            current_stage = "Approved"
        elif current_status == 'Rejected':
            current_stage = "Rejected"
        else:
            current_stage = "Submitted"
    
    # Display current stage
    st.markdown(f"### 📍 **Current Stage:** {current_stage}")
    
    # Workflow steps
    workflow_steps = [
        {
            "step": 1,
            "title": "Document Submission",
            "description": "Upload all required documents through the Document Upload portal",
            "status": "Complete" if len(submissions) > 0 else "Pending",
            "action": "Go to Document Upload tab",
            "color": "green" if len(submissions) > 0 else "gray"
        },
        {
            "step": 2,
            "title": "Data Submission",
            "description": "Submit institutional performance data through data submission forms",
            "status": "Complete" if len(submissions) > 0 else "Pending",
            "action": "Go to Data Submission tabs",
            "color": "green" if len(submissions) > 0 else "gray"
        },
        {
            "step": 3,
            "title": "AI Analysis & Verification",
            "description": "System automatically analyzes documents and data for completeness",
            "status": "In Progress" if len(submissions) > 0 and current_stage == "Submitted" else "Pending",
            "action": "Automatic Process",
            "color": "orange" if len(submissions) > 0 and current_stage == "Submitted" else "gray"
        },
        {
            "step": 4,
            "title": "Committee Review",
            "description": "UGC/AICTE committee reviews AI recommendations and documents",
            "status": "In Progress" if current_stage == "Under Committee Review" else "Pending",
            "action": "Under Committee Review",
            "color": "orange" if current_stage == "Under Committee Review" else "gray"
        },
        {
            "step": 5,
            "title": "Final Decision",
            "description": "Receive final approval decision with conditions and timeline",
            "status": "Complete" if current_stage in ["Approved", "Rejected"] else "Pending",
            "action": "Awaiting Decision" if current_stage not in ["Approved", "Rejected"] else "Decision Made",
            "color": "green" if current_stage in ["Approved", "Rejected"] else "gray"
        }
    ]
    
    # Display workflow
    st.markdown("### 📋 Approval Process Steps")
    
    for step in workflow_steps:
        with st.expander(f"Step {step['step']}: {step['title']} - {step['status']}", 
                        expanded=step['status'] in ["In Progress", "Complete"]):
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**Description:** {step['description']}")
                st.write(f"**Status:** {step['status']}")
            
            with col2:
                # Status indicator
                if step['status'] == "Complete":
                    st.success("✅")
                elif step['status'] == "In Progress":
                    st.warning("🔄")
                else:
                    st.info("⏳")
            
            with col3:
                # Action button
                if step['status'] == "Pending":
                    if st.button(step['action'], key=f"step_{step['step']}", disabled=True):
                        pass
                elif step['action'] == "Go to Document Upload tab":
                    if st.button(step['action'], key=f"step_{step['step']}"):
                        st.session_state.active_tab = "📤 Document Upload"
                        st.rerun()
                elif step['action'] == "Go to Data Submission tabs":
                    if st.button(step['action'], key=f"step_{step['step']}"):
                        st.session_state.active_tab = "📝 Basic Data Submission"
                        st.rerun()
                else:
                    st.button(step['action'], key=f"step_{step['step']}", disabled=True)
    
    # Estimated timeline
    st.markdown("---")
    st.markdown("### ⏱️ Estimated Timeline")
    
    timeline_data = {
        "Stage": ["Document Submission", "Data Submission", "AI Analysis", "Committee Review", "Final Decision"],
        "Duration": ["1-2 weeks", "1 week", "1-2 days", "2-4 weeks", "1 week"],
        "Status": [step['status'] for step in workflow_steps]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    
    # Contact information
    st.markdown("---")
    st.markdown("### 📞 Support & Contact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **For Application Queries:**
        - Email: applications@ugc.gov.in
        - Phone: 011-12345678
        - Hours: 9 AM - 5 PM (Mon-Fri)
        """)
    
    with col2:
        st.info("""
        **Technical Support:**
        - Email: techsupport@sugam.gov.in
        - Phone: 011-87654321
        - Live Chat: Available on portal
        """)
