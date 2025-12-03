import streamlit as st
from datetime import datetime

def create_institution_data_submission(analyzer, user):
    st.subheader("📝 Basic Data Submission Form")
    
    st.info("Submit essential institutional data and performance metrics through this simplified form")
    
    with st.form("basic_institution_data_submission"):
        st.write("### 🎓 Academic Performance Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            naac_grade = st.selectbox(
                "NAAC Grade",
                ["A++", "A+", "A", "B++", "B+", "B", "C", "Not Accredited"],
                key="basic_naac_grade"
            )
            student_faculty_ratio = st.number_input(
                "Student-Faculty Ratio",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=0.1,
                key="basic_sf_ratio"
            )
        
        with col2:
            nirf_ranking = st.number_input(
                "NIRF Ranking (if applicable)",
                min_value=1,
                max_value=200,
                value=None,
                placeholder="Leave blank if not ranked",
                key="basic_nirf"
            )
            placement_rate = st.number_input(
                "Placement Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key="basic_placement"
            )
        
        submitted = st.form_submit_button("📤 Submit Basic Data")
        
        if submitted:
            submission_data = {
                "academic_data": {
                    "naac_grade": naac_grade,
                    "nirf_ranking": nirf_ranking,
                    "student_faculty_ratio": student_faculty_ratio,
                    "placement_rate": placement_rate
                },
                "submission_notes": "Basic data submission",
                "submission_date": datetime.now().isoformat(),
                "submission_type": "basic_institutional_data"
            }
            
            analyzer.save_institution_submission(
                user['institution_id'],
                "basic_performance_data",
                submission_data
            )
            
            st.success("✅ Basic data submitted successfully! Your submission is under review.")
            st.balloons()

def create_systematic_data_submission_form(analyzer, user):
    st.subheader("🏛️ Systematic Data Submission Form - NEP 2020 Framework")
    
    st.info("""
    **Complete this comprehensive data submission form based on the 10-parameter framework from the Dr. Radhakrishnan Committee Report.**
    This data will be used for AI-powered institutional analysis and accreditation assessment.
    """)
    
    with st.form("systematic_data_submission"):
        st.markdown("### 📚 1. CURRICULUM")
        
        curriculum_framework_score = st.slider(
            "Curriculum Framework Quality Score (1-10)",
            min_value=1, max_value=10, value=7,
            help="Assessment of curriculum design and structure"
        )
        
        st.markdown("### 👨‍🏫 2. FACULTY RESOURCES")
        
        faculty_student_ratio = st.number_input(
            "Student-Faculty Ratio",
            min_value=5.0, max_value=50.0, value=20.0, step=0.1
        )
        
        submitted = st.form_submit_button("🚀 Submit Comprehensive Institutional Data")
        
        if submitted:
            submission_data = {
                "submission_type": "comprehensive_institutional_data",
                "submission_date": datetime.now().isoformat(),
                "parameter_scores": {
                    "curriculum": {
                        "curriculum_framework_score": curriculum_framework_score,
                    },
                    "faculty_resources": {
                        "faculty_student_ratio": faculty_student_ratio,
                    }
                }
            }
            
            analyzer.save_institution_submission(
                user['institution_id'],
                "comprehensive_institutional_data",
                submission_data
            )
            
            st.success("✅ Comprehensive Institutional Data Submitted Successfully!")
            st.balloons()
