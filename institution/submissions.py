import streamlit as st
import pandas as pd
import json

def create_institution_submissions_view(analyzer, user):
    st.subheader("📊 My Submissions & Status")
    
    submissions = analyzer.get_institution_submissions(user['institution_id'])
    
    if len(submissions) > 0:
        for _, submission in submissions.iterrows():
            with st.expander(f"{submission['submission_type']} - {submission['submitted_date']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Status:** {submission['status']}")
                with col2:
                    st.write(f"**Submitted:** {submission['submitted_date']}")
                with col3:
                    if submission['reviewed_by']:
                        st.write(f"**Reviewed by:** {submission['reviewed_by']}")
                
                if submission['review_comments']:
                    st.info(f"**Review Comments:** {submission['review_comments']}")
    else:
        st.info("No submissions found. Use the Data Submission tab to submit your institutional data.")
