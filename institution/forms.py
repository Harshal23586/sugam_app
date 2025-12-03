import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def create_institution_data_submission(analyzer, user):
    """Create basic data submission form"""
    st.subheader("📝 Basic Data Submission Form")
    
    st.info("""
    **Submit essential institutional data and performance metrics through this simplified form.**
    This data will be used for preliminary assessment and approval processes.
    """)
    
    with st.form("basic_institution_data_submission", clear_on_submit=True):
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
            phd_faculty_ratio = st.number_input(
                "PhD Faculty Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
                key="basic_phd_ratio"
            ) / 100
        
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
        
        st.write("### 🔬 Research & Innovation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            research_publications = st.number_input(
                "Research Publications (Last 3 Years)",
                min_value=0,
                value=50,
                key="basic_publications"
            )
            research_grants = st.number_input(
                "Research Grants Amount (₹ Lakhs)",
                min_value=0,
                value=100,
                step=10,
                key="basic_grants"
            )
        
        with col2:
            patents_filed = st.number_input(
                "Patents Filed (Last 3 Years)",
                min_value=0,
                value=5,
                key="basic_patents"
            )
            industry_collaborations = st.number_input(
                "Industry Collaborations",
                min_value=0,
                value=8,
                key="basic_industry"
            )
        
        st.write("### 🏗️ Infrastructure & Facilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            digital_infrastructure_score = st.slider(
                "Digital Infrastructure Score (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                key="basic_digital"
            )
            library_volumes = st.number_input(
                "Library Volumes (in thousands)",
                min_value=0,
                value=20,
                key="basic_library"
            )
        
        with col2:
            laboratory_score = st.slider(
                "Laboratory Equipment Score (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                key="basic_lab"
            )
            campus_area = st.number_input(
                "Campus Area (Acres)",
                min_value=0.0,
                value=50.0,
                step=1.0,
                key="basic_campus"
            )
        
        st.write("### ⚖️ Governance & Administration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            financial_stability_score = st.slider(
                "Financial Stability Score (1-10)",
                min_value=1,
                max_value=10,
                value=8,
                key="basic_financial"
            )
            compliance_score = st.slider(
                "Compliance Score (1-10)",
                min_value=1,
                max_value=10,
                value=8,
                key="basic_compliance"
            )
        
        with col2:
            administrative_efficiency = st.slider(
                "Administrative Efficiency (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                key="basic_admin"
            )
            grievance_redressal = st.slider(
                "Grievance Redressal Score (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                key="basic_grievance"
            )
        
        st.write("### 👥 Student Development & Support")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_students = st.number_input(
                "Total Students",
                min_value=0,
                value=1000,
                step=100,
                key="basic_students"
            )
        
        with col2:
            total_faculty = st.number_input(
                "Total Faculty",
                min_value=0,
                value=50,
                key="basic_faculty"
            )
        
        with col3:
            higher_education_rate = st.number_input(
                "Higher Education Progression (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="basic_higher_ed"
            )
        
        st.write("### 🌍 Social Impact & Outreach")
        
        col1, col2 = st.columns(2)
        
        with col1:
            community_projects = st.number_input(
                "Community Engagement Projects (Last Year)",
                min_value=0,
                value=10,
                key="basic_community"
            )
        
        with col2:
            rural_outreach_score = st.slider(
                "Rural Outreach Score (1-10)",
                min_value=1,
                max_value=10,
                value=6,
                key="basic_rural"
            )
        
        # Additional information
        st.write("### 📝 Additional Information")
        
        submission_notes = st.text_area(
            "Additional Notes / Comments",
            placeholder="Add any additional information or context for your submission...",
            height=100,
            key="basic_notes"
        )
        
        # Form submission
        submitted = st.form_submit_button("📤 Submit Basic Data")
        
        if submitted:
            # Validate required fields
            if not all([naac_grade, student_faculty_ratio, placement_rate]):
                st.error("Please fill in all required fields!")
                return
            
            # Prepare submission data
            submission_data = {
                "submission_type": "basic_institutional_data",
                "submission_date": datetime.now().isoformat(),
                "institution_info": {
                    "institution_id": user['institution_id'],
                    "contact_person": user['contact_person'],
                    "email": user['email']
                },
                "academic_data": {
                    "naac_grade": naac_grade,
                    "nirf_ranking": nirf_ranking,
                    "student_faculty_ratio": student_faculty_ratio,
                    "phd_faculty_ratio": phd_faculty_ratio,
                    "placement_rate": placement_rate,
                    "higher_education_rate": higher_education_rate
                },
                "research_data": {
                    "research_publications": research_publications,
                    "research_grants": research_grants,
                    "patents_filed": patents_filed,
                    "industry_collaborations": industry_collaborations
                },
                "infrastructure_data": {
                    "digital_infrastructure_score": digital_infrastructure_score,
                    "library_volumes": library_volumes,
                    "laboratory_score": laboratory_score,
                    "campus_area": campus_area
                },
                "governance_data": {
                    "financial_stability_score": financial_stability_score,
                    "compliance_score": compliance_score,
                    "administrative_efficiency": administrative_efficiency,
                    "grievance_redressal": grievance_redressal
                },
                "student_data": {
                    "total_students": total_students,
                    "total_faculty": total_faculty
                },
                "social_impact_data": {
                    "community_projects": community_projects,
                    "rural_outreach_score": rural_outreach_score
                },
                "submission_notes": submission_notes
            }
            
            # Save to database
            analyzer.save_institution_submission(
                user['institution_id'],
                "basic_institutional_data",
                submission_data
            )
            
            # Show success message
            st.success("✅ Basic data submitted successfully! Your submission is under review.")
            
            # Show performance preview
            show_performance_preview(submission_data, analyzer)
            
            # Provide next steps
            st.info("""
            **Next Steps:**
            1. Your data will be analyzed by our AI system
            2. You will receive an email confirmation
            3. Check the 'My Submissions' tab for updates
            4. Consider submitting the comprehensive Systematic Data Form for detailed assessment
            """)
            
            st.balloons()

def create_systematic_data_submission_form(analyzer, user):
    """Create comprehensive systematic data submission form based on NEP 2020 framework"""
    st.subheader("🏛️ Systematic Data Submission Form - NEP 2020 Framework")
    
    st.info("""
    **Complete this comprehensive data submission form based on the 10-parameter framework from the Dr. Radhakrishnan Committee Report.**
    
    This detailed data will be used for:
    - AI-powered institutional analysis
    - Accreditation assessment
    - Performance benchmarking
    - Strategic planning and recommendations
    """)
    
    # Progress tracking
    st.markdown("### 📋 Form Progress")
    progress_placeholder = st.empty()
    
    with st.form("systematic_data_submission", clear_on_submit=True):
        # Initialize session state for progress tracking
        if 'form_progress' not in st.session_state:
            st.session_state.form_progress = 0
        
        # 1. CURRICULUM
        with st.expander("📚 1. CURRICULUM", expanded=True):
            st.markdown("#### Curriculum Design & Development")
            
            col1, col2 = st.columns(2)
            
            with col1:
                curriculum_framework_score = st.slider(
                    "Curriculum Framework Quality Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    help="Assessment of curriculum design and structure",
                    key="curriculum_framework"
                )
                stakeholder_consultation = st.selectbox(
                    "Stakeholder Consultation in Curriculum Design",
                    ["Regular & Comprehensive", "Occasional", "Minimal", "None"],
                    help="Industry, alumni, employer involvement",
                    key="stakeholder_consult"
                )
                curriculum_update_frequency = st.selectbox(
                    "Curriculum Review & Update Frequency",
                    ["Annual", "Biannual", "Every 3 Years", "Irregular"],
                    help="How often curriculum is revised",
                    key="curriculum_update"
                )
            
            with col2:
                multidisciplinary_courses = st.number_input(
                    "Number of Multidisciplinary Courses",
                    min_value=0, value=5,
                    help="Courses integrating multiple disciplines",
                    key="multidisciplinary"
                )
                skill_integration_score = st.slider(
                    "Skill Integration in Curriculum (1-10)",
                    min_value=1, max_value=10, value=6,
                    help="Integration of vocational and employability skills",
                    key="skill_integration"
                )
                digital_content_availability = st.selectbox(
                    "Digital Learning Content Availability",
                    ["Extensive (>80%)", "Moderate (50-80%)", "Limited (<50%)", "Minimal"],
                    help="Availability of digital learning materials",
                    key="digital_content"
                )
            
            # Learning Outcomes
            st.markdown("#### Learning Outcomes Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                learning_outcome_achievement = st.number_input(
                    "Learning Outcome Achievement Rate (%)",
                    min_value=0.0, max_value=100.0, value=75.0, step=1.0,
                    help="Percentage of learning outcomes achieved",
                    key="learning_outcomes"
                )
                assessment_methods = st.multiselect(
                    "Assessment Methods Used",
                    ["Written Exams", "Practical Exams", "Projects", "Presentations", 
                     "Portfolios", "Peer Assessment", "Self Assessment", "Online Quizzes"],
                    default=["Written Exams", "Projects", "Presentations"],
                    key="assessment_methods"
                )
            
            with col2:
                critical_thinking_assessment = st.selectbox(
                    "Critical Thinking Assessment Integration",
                    ["Comprehensive", "Moderate", "Limited", "None"],
                    help="Assessment of analytical and critical thinking skills",
                    key="critical_thinking"
                )
                feedback_mechanism = st.selectbox(
                    "Student Feedback Collection Mechanism",
                    ["Regular & Systematic", "Occasional", "Ad-hoc", "None"],
                    help="System for collecting student feedback on curriculum",
                    key="feedback_mechanism"
                )
        
        # 2. FACULTY RESOURCES
        with st.expander("👨‍🏫 2. FACULTY RESOURCES", expanded=False):
            st.markdown("#### Faculty Composition & Qualifications")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                faculty_student_ratio = st.number_input(
                    "Student-Faculty Ratio",
                    min_value=5.0, max_value=50.0, value=20.0, step=0.1,
                    key="faculty_ratio"
                )
                phd_faculty_percentage = st.number_input(
                    "PhD Faculty Percentage (%)",
                    min_value=0.0, max_value=100.0, value=65.0, step=1.0,
                    key="phd_percentage"
                )
                faculty_diversity_index = st.slider(
                    "Faculty Diversity Index (1-10)",
                    min_value=1, max_value=10, value=6,
                    help="Gender, social, regional diversity",
                    key="faculty_diversity"
                )
            
            with col2:
                international_faculty = st.number_input(
                    "International Faculty Percentage (%)",
                    min_value=0.0, max_value=100.0, value=5.0, step=1.0,
                    key="international_faculty"
                )
                industry_exposure_faculty = st.number_input(
                    "Faculty with Industry Exposure (%)",
                    min_value=0.0, max_value=100.0, value=30.0, step=1.0,
                    key="industry_exposure"
                )
                faculty_retention_rate = st.number_input(
                    "Faculty Retention Rate (%)",
                    min_value=0.0, max_value=100.0, value=85.0, step=1.0,
                    key="faculty_retention"
                )
            
            with col3:
                faculty_development_hours = st.number_input(
                    "Annual Faculty Development Hours per Faculty",
                    min_value=0, value=40,
                    help="Training and development hours",
                    key="faculty_dev_hours"
                )
                research_publications_per_faculty = st.number_input(
                    "Research Publications per Faculty (Annual)",
                    min_value=0.0, value=1.5, step=0.1,
                    key="pubs_per_faculty"
                )
                faculty_awards = st.number_input(
                    "Faculty Awards & Recognitions (Last 3 Years)",
                    min_value=0, value=10,
                    key="faculty_awards"
                )
            
            # Faculty Development Programs
            st.markdown("#### Faculty Development & Support")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fdp_programs = st.number_input(
                    "Faculty Development Programs Conducted",
                    min_value=0, value=8,
                    help="Number of FDPs conducted annually",
                    key="fdp_programs"
                )
                mentorship_programs = st.selectbox(
                    "Faculty Mentorship Programs",
                    ["Comprehensive", "Moderate", "Limited", "None"],
                    key="mentorship"
                )
            
            with col2:
                teaching_innovations = st.number_input(
                    "Teaching Innovations Implemented",
                    min_value=0, value=5,
                    help="Number of innovative teaching methods adopted",
                    key="teaching_innovations"
                )
                faculty_feedback = st.slider(
                    "Faculty Satisfaction Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="faculty_satisfaction"
                )
        
        # 3. LEARNING AND TEACHING
        with st.expander("🎓 3. LEARNING AND TEACHING", expanded=False):
            st.markdown("#### Teaching-Learning Process")
            
            col1, col2 = st.columns(2)
            
            with col1:
                average_attendance_rate = st.number_input(
                    "Average Student Attendance Rate (%)",
                    min_value=0.0, max_value=100.0, value=85.0, step=1.0,
                    key="attendance_rate"
                )
                digital_platform_usage = st.selectbox(
                    "Digital Platform Usage in Teaching",
                    ["Extensive Integration", "Moderate Use", "Limited Use", "Traditional Only"],
                    help="Use of LMS, online tools, digital resources",
                    key="digital_platform"
                )
                experiential_learning_hours = st.number_input(
                    "Experiential Learning Hours per Student (Annual)",
                    min_value=0, value=50,
                    key="experiential_learning"
                )
            
            with col2:
                student_feedback_score = st.slider(
                    "Student Feedback Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="student_feedback"
                )
                teaching_methodologies = st.multiselect(
                    "Teaching Methodologies Used",
                    ["Lecture-based", "Case Studies", "Project-based", "Flipped Classroom",
                     "Collaborative Learning", "Problem-based Learning", "Online Blended"],
                    default=["Lecture-based", "Case Studies", "Project-based"],
                    key="teaching_methods"
                )
                library_usage = st.number_input(
                    "Library Usage per Student (Hours/Week)",
                    min_value=0.0, value=5.0, step=0.5,
                    key="library_usage"
                )
            
            # Student Support
            st.markdown("#### Student Support Services")
            
            col1, col2 = st.columns(2)
            
            with col1:
                academic_counseling = st.selectbox(
                    "Academic Counseling System",
                    ["Comprehensive", "Moderate", "Limited", "None"],
                    key="academic_counseling"
                )
                remedial_classes = st.selectbox(
                    "Remedial Classes for Weak Students",
                    ["Regular", "Occasional", "On-demand", "None"],
                    key="remedial_classes"
                )
            
            with col2:
                student_grievance_redressal = st.selectbox(
                    "Student Grievance Redressal Mechanism",
                    ["Effective & Timely", "Moderate", "Slow", "Ineffective"],
                    key="student_grievance"
                )
                peer_learning = st.selectbox(
                    "Peer Learning & Tutoring Programs",
                    ["Comprehensive", "Moderate", "Limited", "None"],
                    key="peer_learning"
                )
        
        # 4. RESEARCH AND INNOVATION
        with st.expander("🔬 4. RESEARCH AND INNOVATION", expanded=False):
            st.markdown("#### Research Output & Quality")
            
            col1, col2 = st.columns(2)
            
            with col1:
                research_publications_total = st.number_input(
                    "Total Research Publications (Last 3 Years)",
                    min_value=0, value=100,
                    key="total_publications"
                )
                scopus_indexed = st.number_input(
                    "Scopus Indexed Publications",
                    min_value=0, value=40,
                    key="scopus_pubs"
                )
                patents_filed = st.number_input(
                    "Patents Filed (Last 3 Years)",
                    min_value=0, value=10,
                    key="patents_total"
                )
                h_index_institution = st.number_input(
                    "Institutional H-index",
                    min_value=0, value=25,
                    key="h_index"
                )
            
            with col2:
                research_grants_amount = st.number_input(
                    "Research Grants Received (₹ Lakhs - Last 3 Years)",
                    min_value=0, value=500,
                    key="research_grants"
                )
                industry_collaborations = st.number_input(
                    "Industry Research Collaborations",
                    min_value=0, value=8,
                    key="industry_collabs"
                )
                international_research_partnerships = st.number_input(
                    "International Research Partnerships",
                    min_value=0, value=5,
                    key="international_partnerships"
                )
                research_facility_utilization = st.slider(
                    "Research Facility Utilization Rate (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="research_facility"
                )
            
            # Research Support
            st.markdown("#### Research Support & Culture")
            
            col1, col2 = st.columns(2)
            
            with col1:
                student_research_participation = st.number_input(
                    "Student Research Participation (%)",
                    min_value=0.0, max_value=100.0, value=40.0, step=1.0,
                    key="student_research"
                )
                research_seminars = st.number_input(
                    "Research Seminars/Conferences Organized",
                    min_value=0, value=12,
                    key="research_seminars"
                )
            
            with col2:
                ipr_cell = st.selectbox(
                    "IPR Cell/Technology Transfer Office",
                    ["Functional & Active", "Exists but Limited", "In Planning", "None"],
                    key="ipr_cell"
                )
                startup_incubation = st.selectbox(
                    "Startup Incubation/Entrepreneurship Cell",
                    ["Functional & Active", "Exists but Limited", "In Planning", "None"],
                    key="startup_incubation"
                )
        
        # 5. EXTRACURRICULAR & CO-CURRICULAR ACTIVITIES
        with st.expander("⚽ 5. EXTRACURRICULAR & CO-CURRICULAR ACTIVITIES", expanded=False):
            st.markdown("#### Student Activities & Engagement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ec_activities_annual = st.number_input(
                    "Annual Extracurricular Activities",
                    min_value=0, value=25,
                    key="ec_activities"
                )
                student_participation_rate_ec = st.number_input(
                    "Student Participation Rate in EC Activities (%)",
                    min_value=0.0, max_value=100.0, value=60.0, step=1.0,
                    key="ec_participation"
                )
                sports_infrastructure_score = st.slider(
                    "Sports Infrastructure Quality (1-10)",
                    min_value=1, max_value=10, value=6,
                    key="sports_infra"
                )
            
            with col2:
                cultural_events_annual = st.number_input(
                    "Annual Cultural Events",
                    min_value=0, value=15,
                    key="cultural_events"
                )
                leadership_programs = st.number_input(
                    "Leadership Development Programs",
                    min_value=0, value=8,
                    key="leadership_programs"
                )
                ec_credit_integration = st.selectbox(
                    "EC/CC Credit Integration in Curriculum",
                    ["Fully Integrated", "Partially Integrated", "Separate", "None"],
                    help="Integration of extracurricular credits",
                    key="ec_credits"
                )
            
            # Clubs & Societies
            st.markdown("#### Student Clubs & Societies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                student_clubs = st.number_input(
                    "Active Student Clubs/Societies",
                    min_value=0, value=15,
                    key="student_clubs"
                )
                club_funding = st.selectbox(
                    "Club Funding & Support",
                    ["Adequate", "Moderate", "Limited", "Insufficient"],
                    key="club_funding"
                )
            
            with col2:
                national_level_participation = st.number_input(
                    "National Level Competition Participation",
                    min_value=0, value=20,
                    key="national_participation"
                )
                achievement_awards = st.number_input(
                    "Awards/Achievements in EC/CC Activities",
                    min_value=0, value=30,
                    key="ec_awards"
                )
        
        # 6. COMMUNITY ENGAGEMENT
        with st.expander("🤝 6. COMMUNITY ENGAGEMENT", expanded=False):
            st.markdown("#### Social Outreach & Engagement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                community_projects_annual = st.number_input(
                    "Annual Community Engagement Projects",
                    min_value=0, value=12,
                    key="community_projects"
                )
                student_participation_community = st.number_input(
                    "Student Participation in Community Service (%)",
                    min_value=0.0, max_value=100.0, value=45.0, step=1.0,
                    key="community_participation"
                )
                rural_outreach_programs = st.number_input(
                    "Rural Outreach Programs",
                    min_value=0, value=6,
                    key="rural_outreach"
                )
            
            with col2:
                social_impact_assessment = st.selectbox(
                    "Social Impact Assessment Conducted",
                    ["Regularly", "Occasionally", "Rarely", "Never"],
                    key="social_impact"
                )
                csr_initiatives = st.number_input(
                    "CSR Initiatives Undertaken",
                    min_value=0, value=4,
                    key="csr_initiatives"
                )
                community_feedback_score = st.slider(
                    "Community Feedback Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="community_feedback"
                )
            
            # NSS/NCC Activities
            st.markdown("#### NSS/NCC & Extension Activities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nss_ncc_participation = st.number_input(
                    "NSS/NCC Student Participation (%)",
                    min_value=0.0, max_value=100.0, value=30.0, step=1.0,
                    key="nss_participation"
                )
                extension_activities = st.number_input(
                    "Extension Activities (Last Year)",
                    min_value=0, value=10,
                    key="extension_activities"
                )
            
            with col2:
                village_adoption = st.selectbox(
                    "Village/Community Adoption Program",
                    ["Active & Ongoing", "Occasional", "One-time", "None"],
                    key="village_adoption"
                )
                disaster_relief = st.selectbox(
                    "Disaster Relief/Rehabilitation Participation",
                    ["Active Participation", "Occasional", "Limited", "None"],
                    key="disaster_relief"
                )
        
        # 7. GREEN INITIATIVES
        with st.expander("🌱 7. GREEN INITIATIVES", expanded=False):
            st.markdown("#### Environmental Sustainability")
            
            col1, col2 = st.columns(2)
            
            with col1:
                renewable_energy_usage = st.number_input(
                    "Renewable Energy Usage (%)",
                    min_value=0.0, max_value=100.0, value=25.0, step=1.0,
                    key="renewable_energy"
                )
                waste_management_score = st.slider(
                    "Waste Management System Score (1-10)",
                    min_value=1, max_value=10, value=6,
                    key="waste_management"
                )
                water_harvesting_capacity = st.number_input(
                    "Water Harvesting Capacity (KL Annual)",
                    min_value=0, value=5000,
                    key="water_harvesting"
                )
            
            with col2:
                carbon_footprint_reduction = st.number_input(
                    "Carbon Footprint Reduction (%) - Last 3 Years",
                    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                    key="carbon_reduction"
                )
                green_cover_percentage = st.number_input(
                    "Green Cover Percentage on Campus",
                    min_value=0.0, max_value=100.0, value=40.0, step=1.0,
                    key="green_cover"
                )
                environmental_awareness_programs = st.number_input(
                    "Environmental Awareness Programs (Annual)",
                    min_value=0, value=10,
                    key="env_awareness"
                )
            
            # Green Campus Initiatives
            st.markdown("#### Green Campus Initiatives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                green_buildings = st.selectbox(
                    "Green Building Certification",
                    ["Platinum", "Gold", "Silver", "In Progress", "None"],
                    key="green_buildings"
                )
                e_waste_management = st.selectbox(
                    "E-waste Management System",
                    ["Comprehensive", "Moderate", "Basic", "None"],
                    key="e_waste"
                )
            
            with col2:
                biodiversity_conservation = st.selectbox(
                    "Biodiversity Conservation Efforts",
                    ["Active Program", "Moderate", "Limited", "None"],
                    key="biodiversity"
                )
                sustainable_transport = st.selectbox(
                    "Sustainable Transport Initiatives",
                    ["Comprehensive", "Moderate", "Limited", "None"],
                    key="sustainable_transport"
                )
        
        # 8. GOVERNANCE AND ADMINISTRATION
        with st.expander("⚖️ 8. GOVERNANCE AND ADMINISTRATION", expanded=False):
            st.markdown("#### Institutional Governance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                governance_transparency_score = st.slider(
                    "Governance Transparency Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="governance_transparency"
                )
                grievance_redressal_time = st.number_input(
                    "Average Grievance Redressal Time (Days)",
                    min_value=1, value=15,
                    key="grievance_time"
                )
                egov_implementation_level = st.selectbox(
                    "e-Governance Implementation Level",
                    ["Advanced", "Moderate", "Basic", "Minimal"],
                    key="egov_level"
                )
            
            with col2:
                student_participation_governance = st.number_input(
                    "Student Participation in Governance (%)",
                    min_value=0.0, max_value=100.0, value=30.0, step=1.0,
                    key="student_governance"
                )
                administrative_efficiency_score = st.slider(
                    "Administrative Efficiency Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="admin_efficiency"
                )
                international_collaborations = st.number_input(
                    "Active International Collaborations",
                    min_value=0, value=8,
                    key="intl_collabs"
                )
            
            # Strategic Planning
            st.markdown("#### Strategic Planning & Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategic_plan = st.selectbox(
                    "Strategic Plan Implementation",
                    ["Comprehensive & Active", "Moderate", "Basic", "None"],
                    key="strategic_plan"
                )
                stakeholder_engagement = st.selectbox(
                    "Stakeholder Engagement in Decision Making",
                    ["Regular & Meaningful", "Occasional", "Limited", "None"],
                    key="stakeholder_engagement"
                )
            
            with col2:
                risk_management = st.selectbox(
                    "Risk Management Framework",
                    ["Comprehensive", "Moderate", "Basic", "None"],
                    key="risk_management"
                )
                audit_compliance = st.selectbox(
                    "Audit Compliance Status",
                    ["Clean Reports", "Minor Issues", "Major Issues", "Pending"],
                    key="audit_compliance"
                )
        
        # 9. INFRASTRUCTURE DEVELOPMENT
        with st.expander("🏗️ 9. INFRASTRUCTURE DEVELOPMENT", expanded=False):
            st.markdown("#### Physical Infrastructure")
            
            col1, col2 = st.columns(2)
            
            with col1:
                campus_area = st.number_input(
                    "Campus Area (Acres)",
                    min_value=0.0, value=50.0, step=0.1,
                    key="campus_area"
                )
                digital_infrastructure_score = st.slider(
                    "Digital Infrastructure Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="digital_infra"
                )
                laboratory_equipment_score = st.slider(
                    "Laboratory Equipment Quality Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="lab_equipment"
                )
            
            with col2:
                library_resources_score = st.slider(
                    "Library Resources Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="library_resources"
                )
                hostel_capacity_utilization = st.number_input(
                    "Hostel Capacity Utilization (%)",
                    min_value=0.0, max_value=100.0, value=80.0, step=1.0,
                    key="hostel_utilization"
                )
                infrastructure_maintenance_score = st.slider(
                    "Infrastructure Maintenance Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="infra_maintenance"
                )
            
            # Special Facilities
            st.markdown("#### Specialized Facilities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sports_complex = st.selectbox(
                    "Sports Complex Facilities",
                    ["Comprehensive", "Moderate", "Basic", "None"],
                    key="sports_complex"
                )
                auditorium_capacity = st.number_input(
                    "Auditorium Capacity",
                    min_value=0, value=500,
                    key="auditorium_capacity"
                )
            
            with col2:
                disabled_friendly = st.selectbox(
                    "Disabled-friendly Infrastructure",
                    ["Fully Accessible", "Partially Accessible", "Limited", "Inaccessible"],
                    key="disabled_friendly"
                )
                smart_classrooms = st.number_input(
                    "Smart Classrooms",
                    min_value=0, value=20,
                    key="smart_classrooms"
                )
        
        # 10. FINANCIAL RESOURCES AND MANAGEMENT
        with st.expander("💰 10. FINANCIAL RESOURCES AND MANAGEMENT", expanded=False):
            st.markdown("#### Financial Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                financial_stability_score = st.slider(
                    "Financial Stability Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="financial_stability"
                )
                research_investment_percentage = st.number_input(
                    "Research Investment (% of Total Budget)",
                    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                    key="research_investment"
                )
                infrastructure_investment = st.number_input(
                    "Infrastructure Investment (₹ Lakhs - Annual)",
                    min_value=0, value=200,
                    key="infra_investment"
                )
            
            with col2:
                revenue_generation_score = st.slider(
                    "Revenue Generation Score (1-10)",
                    min_value=1, max_value=10, value=6,
                    key="revenue_generation"
                )
                financial_aid_students = st.number_input(
                    "Students Receiving Financial Aid (%)",
                    min_value=0.0, max_value=100.0, value=25.0, step=1.0,
                    key="financial_aid"
                )
                audit_compliance_score = st.slider(
                    "Audit Compliance Score (1-10)",
                    min_value=1, max_value=10, value=8,
                    key="audit_score"
                )
            
            # Resource Mobilization
            st.markdown("#### Resource Mobilization & Utilization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                endowment_funds = st.number_input(
                    "Endowment Funds (₹ Crores)",
                    min_value=0.0, value=10.0, step=0.1,
                    key="endowment_funds"
                )
                industry_funding = st.number_input(
                    "Industry Funding (₹ Lakhs - Annual)",
                    min_value=0, value=100,
                    key="industry_funding"
                )
            
            with col2:
                grant_utilization = st.selectbox(
                    "Grant Utilization Efficiency",
                    ["Excellent (>90%)", "Good (75-90%)", "Average (60-75%)", "Poor (<60%)"],
                    key="grant_utilization"
                )
                budget_discipline = st.selectbox(
                    "Budget Discipline & Control",
                    ["Excellent", "Good", "Average", "Poor"],
                    key="budget_discipline"
                )
        
        # ADDITIONAL INSTITUTIONAL METRICS
        with st.expander("📊 ADDITIONAL INSTITUTIONAL METRICS", expanded=False):
            st.markdown("#### Overall Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                placement_rate = st.number_input(
                    "Placement Rate (%)",
                    min_value=0.0, max_value=100.0, value=75.0, step=1.0,
                    key="placement_rate"
                )
                higher_education_rate = st.number_input(
                    "Higher Education Progression Rate (%)",
                    min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                    key="higher_ed_rate"
                )
            
            with col2:
                entrepreneurship_cell_score = st.slider(
                    "Entrepreneurship Cell Activity Score (1-10)",
                    min_value=1, max_value=10, value=6,
                    key="entrepreneurship_score"
                )
                alumni_engagement_score = st.slider(
                    "Alumni Engagement Score (1-10)",
                    min_value=1, max_value=10, value=5,
                    key="alumni_engagement"
                )
            
            with col3:
                institutional_reputation_score = st.slider(
                    "Institutional Reputation Score (1-10)",
                    min_value=1, max_value=10, value=7,
                    key="institutional_reputation"
                )
                naac_previous_grade = st.selectbox(
                    "Previous NAAC Grade (if any)",
                    ["A++", "A+", "A", "B++", "B+", "B", "C", "Not Accredited"],
                    key="naac_previous"
                )
        
        # QUALITATIVE INFORMATION
        with st.expander("📝 QUALITATIVE INFORMATION", expanded=False):
            st.markdown("#### Institutional Narrative")
            
            institutional_strengths = st.text_area(
                "Key Institutional Strengths & Achievements",
                placeholder="Describe your institution's major strengths, unique features, and significant achievements...",
                height=100,
                key="institutional_strengths"
            )
            
            improvement_areas = st.text_area(
                "Areas for Improvement & Challenges",
                placeholder="Identify key areas where your institution seeks improvement and challenges faced...",
                height=100,
                key="improvement_areas"
            )
            
            strategic_initiatives = st.text_area(
                "Strategic Initiatives & Future Plans",
                placeholder="Describe ongoing or planned strategic initiatives, expansion plans, and future vision...",
                height=100,
                key="strategic_initiatives"
            )
            
            innovation_initiatives = st.text_area(
                "Innovation & Best Practices",
                placeholder="Describe innovative practices, pedagogical innovations, and institutional best practices...",
                height=100,
                key="innovation_initiatives"
            )
        
        # Progress indicator
        progress_placeholder.progress(st.session_state.form_progress)
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Comprehensive Institutional Data")
        
        if submitted:
            # Validate required fields
            required_fields = [
                curriculum_framework_score, faculty_student_ratio, 
                average_attendance_rate, research_publications_total,
                ec_activities_annual, community_projects_annual,
                renewable_energy_usage, governance_transparency_score,
                campus_area, financial_stability_score, placement_rate
            ]
            
            if any(field == 0 for field in required_fields if isinstance(field, (int, float))):
                st.error("Please fill all required fields!")
                return
            
            # Compile all data into a structured format
            submission_data = compile_submission_data(locals(), user)
            
            # Save to database
            analyzer.save_institution_submission(
                user['institution_id'],
                "comprehensive_institutional_data",
                submission_data
            )
            
            # Show success message and analysis
            show_submission_success(submission_data, analyzer)
            
            # Update progress
            st.session_state.form_progress = 100

def compile_submission_data(form_data, user):
    """Compile form data into structured submission format"""
    return {
        "submission_type": "comprehensive_institutional_data",
        "submission_date": datetime.now().isoformat(),
        "institution_info": {
            "institution_id": user['institution_id'],
            "contact_person": user['contact_person'],
            "email": user['email'],
            "submission_timestamp": datetime.now().isoformat()
        },
        "parameter_scores": {
            "curriculum": {
                "curriculum_framework_score": form_data.get('curriculum_framework_score', 0),
                "stakeholder_consultation": form_data.get('stakeholder_consultation', ''),
                "curriculum_update_frequency": form_data.get('curriculum_update_frequency', ''),
                "multidisciplinary_courses": form_data.get('multidisciplinary_courses', 0),
                "skill_integration_score": form_data.get('skill_integration_score', 0),
                "digital_content_availability": form_data.get('digital_content_availability', ''),
                "learning_outcome_achievement": form_data.get('learning_outcome_achievement', 0),
                "assessment_methods": form_data.get('assessment_methods', []),
                "critical_thinking_assessment": form_data.get('critical_thinking_assessment', ''),
                "feedback_mechanism": form_data.get('feedback_mechanism', '')
            },
            "faculty_resources": {
                "faculty_student_ratio": form_data.get('faculty_student_ratio', 0),
                "phd_faculty_percentage": form_data.get('phd_faculty_percentage', 0),
                "faculty_diversity_index": form_data.get('faculty_diversity_index', 0),
                "international_faculty": form_data.get('international_faculty', 0),
                "industry_exposure_faculty": form_data.get('industry_exposure_faculty', 0),
                "faculty_retention_rate": form_data.get('faculty_retention_rate', 0),
                "faculty_development_hours": form_data.get('faculty_development_hours', 0),
                "research_publications_per_faculty": form_data.get('research_publications_per_faculty', 0),
                "faculty_awards": form_data.get('faculty_awards', 0),
                "fdp_programs": form_data.get('fdp_programs', 0),
                "mentorship_programs": form_data.get('mentorship_programs', ''),
                "teaching_innovations": form_data.get('teaching_innovations', 0),
                "faculty_feedback": form_data.get('faculty_feedback', 0)
            },
            "learning_teaching": {
                "average_attendance_rate": form_data.get('average_attendance_rate', 0),
                "digital_platform_usage": form_data.get('digital_platform_usage', ''),
                "experiential_learning_hours": form_data.get('experiential_learning_hours', 0),
                "student_feedback_score": form_data.get('student_feedback_score', 0),
                "teaching_methodologies": form_data.get('teaching_methodologies', []),
                "library_usage": form_data.get('library_usage', 0),
                "academic_counseling": form_data.get('academic_counseling', ''),
                "remedial_classes": form_data.get('remedial_classes', ''),
                "student_grievance_redressal": form_data.get('student_grievance_redressal', ''),
                "peer_learning": form_data.get('peer_learning', '')
            },
            "research_innovation": {
                "research_publications_total": form_data.get('research_publications_total', 0),
                "scopus_indexed": form_data.get('scopus_indexed', 0),
                "patents_filed": form_data.get('patents_filed', 0),
                "h_index_institution": form_data.get('h_index_institution', 0),
                "research_grants_amount": form_data.get('research_grants_amount', 0),
                "industry_collaborations": form_data.get('industry_collaborations', 0),
                "international_research_partnerships": form_data.get('international_research_partnerships', 0),
                "research_facility_utilization": form_data.get('research_facility_utilization', 0),
                "student_research_participation": form_data.get('student_research_participation', 0),
                "research_seminars": form_data.get('research_seminars', 0),
                "ipr_cell": form_data.get('ipr_cell', ''),
                "startup_incubation": form_data.get('startup_incubation', '')
            },
            "extracurricular_activities": {
                "ec_activities_annual": form_data.get('ec_activities_annual', 0),
                "student_participation_rate_ec": form_data.get('student_participation_rate_ec', 0),
                "sports_infrastructure_score": form_data.get('sports_infrastructure_score', 0),
                "cultural_events_annual": form_data.get('cultural_events_annual', 0),
                "leadership_programs": form_data.get('leadership_programs', 0),
                "ec_credit_integration": form_data.get('ec_credit_integration', ''),
                "student_clubs": form_data.get('student_clubs', 0),
                "club_funding": form_data.get('club_funding', ''),
                "national_level_participation": form_data.get('national_level_participation', 0),
                "achievement_awards": form_data.get('achievement_awards', 0)
            },
            "community_engagement": {
                "community_projects_annual": form_data.get('community_projects_annual', 0),
                "student_participation_community": form_data.get('student_participation_community', 0),
                "rural_outreach_programs": form_data.get('rural_outreach_programs', 0),
                "social_impact_assessment": form_data.get('social_impact_assessment', ''),
                "csr_initiatives": form_data.get('csr_initiatives', 0),
                "community_feedback_score": form_data.get('community_feedback_score', 0),
                "nss_ncc_participation": form_data.get('nss_ncc_participation', 0),
                "extension_activities": form_data.get('extension_activities', 0),
                "village_adoption": form_data.get('village_adoption', ''),
                "disaster_relief": form_data.get('disaster_relief', '')
            },
            "green_initiatives": {
                "renewable_energy_usage": form_data.get('renewable_energy_usage', 0),
                "waste_management_score": form_data.get('waste_management_score', 0),
                "water_harvesting_capacity": form_data.get('water_harvesting_capacity', 0),
                "carbon_footprint_reduction": form_data.get('carbon_footprint_reduction', 0),
                "green_cover_percentage": form_data.get('green_cover_percentage', 0),
                "environmental_awareness_programs": form_data.get('environmental_awareness_programs', 0),
                "green_buildings": form_data.get('green_buildings', ''),
                "e_waste_management": form_data.get('e_waste_management', ''),
                "biodiversity_conservation": form_data.get('biodiversity_conservation', ''),
                "sustainable_transport": form_data.get('sustainable_transport', '')
            },
            "governance_administration": {
                "governance_transparency_score": form_data.get('governance_transparency_score', 0),
                "grievance_redressal_time": form_data.get('grievance_redressal_time', 0),
                "egov_implementation_level": form_data.get('egov_implementation_level', ''),
                "student_participation_governance": form_data.get('student_participation_governance', 0),
                "administrative_efficiency_score": form_data.get('administrative_efficiency_score', 0),
                "international_collaborations": form_data.get('international_collaborations', 0),
                "strategic_plan": form_data.get('strategic_plan', ''),
                "stakeholder_engagement": form_data.get('stakeholder_engagement', ''),
                "risk_management": form_data.get('risk_management', ''),
                "audit_compliance": form_data.get('audit_compliance', '')
            },
            "infrastructure_development": {
                "campus_area": form_data.get('campus_area', 0),
                "digital_infrastructure_score": form_data.get('digital_infrastructure_score', 0),
                "laboratory_equipment_score": form_data.get('laboratory_equipment_score', 0),
                "library_resources_score": form_data.get('library_resources_score', 0),
                "hostel_capacity_utilization": form_data.get('hostel_capacity_utilization', 0),
                "infrastructure_maintenance_score": form_data.get('infrastructure_maintenance_score', 0),
                "sports_complex": form_data.get('sports_complex', ''),
                "auditorium_capacity": form_data.get('auditorium_capacity', 0),
                "disabled_friendly": form_data.get('disabled_friendly', ''),
                "smart_classrooms": form_data.get('smart_classrooms', 0)
            },
            "financial_management": {
                "financial_stability_score": form_data.get('financial_stability_score', 0),
                "research_investment_percentage": form_data.get('research_investment_percentage', 0),
                "infrastructure_investment": form_data.get('infrastructure_investment', 0),
                "revenue_generation_score": form_data.get('revenue_generation_score', 0),
                "financial_aid_students": form_data.get('financial_aid_students', 0),
                "audit_compliance_score": form_data.get('audit_compliance_score', 0),
                "endowment_funds": form_data.get('endowment_funds', 0),
                "industry_funding": form_data.get('industry_funding', 0),
                "grant_utilization": form_data.get('grant_utilization', ''),
                "budget_discipline": form_data.get('budget_discipline', '')
            }
        },
        "additional_metrics": {
            "placement_rate": form_data.get('placement_rate', 0),
            "higher_education_rate": form_data.get('higher_education_rate', 0),
            "entrepreneurship_cell_score": form_data.get('entrepreneurship_cell_score', 0),
            "alumni_engagement_score": form_data.get('alumni_engagement_score', 0),
            "institutional_reputation_score": form_data.get('institutional_reputation_score', 0),
            "naac_previous_grade": form_data.get('naac_previous_grade', '')
        },
        "qualitative_data": {
            "institutional_strengths": form_data.get('institutional_strengths', ''),
            "improvement_areas": form_data.get('improvement_areas', ''),
            "strategic_initiatives": form_data.get('strategic_initiatives', ''),
            "innovation_initiatives": form_data.get('innovation_initiatives', '')
        }
    }

def show_submission_success(submission_data, analyzer):
    """Show success message and analysis after submission"""
    st.success("✅ Comprehensive Institutional Data Submitted Successfully!")
    st.balloons()
    
    # Show submission summary
    st.subheader("📋 Submission Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Parameters Covered", "10")
        st.metric("Data Points Collected", "65+")
    
    with col2:
        st.metric("AI Analysis Ready", "Yes")
        st.metric("NEP 2020 Compliant", "Yes")
    
    with col3:
        st.metric("Submission Status", "Complete")
        st.metric("Submission Time", datetime.now().strftime("%H:%M"))
    
    # Show parameter scores visualization
    st.subheader("📊 Parameter-wise Performance")
    
    parameter_scores = submission_data['parameter_scores']
    avg_scores = {}
    
    for param, metrics in parameter_scores.items():
        # Calculate average score for each parameter
        numeric_values = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric_values.append(value)
        
        if numeric_values:
            avg_scores[param.replace('_', ' ').title()] = sum(numeric_values) / len(numeric_values)
    
    if avg_scores:
        # Create bar chart
        df_scores = pd.DataFrame({
            'Parameter': list(avg_scores.keys()),
            'Average Score': list(avg_scores.values())
        })
        
        fig = px.bar(df_scores, x='Parameter', y='Average Score', 
                    title="Average Scores by Parameter",
                    color='Average Score',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show recommendations
    st.subheader("🤖 Preliminary AI Analysis")
    
    recommendations = [
        "✅ Data submission complete and ready for analysis",
        "📈 Performance benchmarking will be conducted against peer institutions",
        "🎯 Personalized improvement recommendations will be generated",
        "📊 Detailed analytical report will be available in 24-48 hours"
    ]
    
    for rec in recommendations:
        st.info(f"• {rec}")
    
    st.info("""
    **Next Steps:**
    1. Your data will be processed for AI-powered institutional analysis
    2. Comprehensive assessment report will be generated
    3. Accreditation recommendations will be provided
    4. You can track the analysis progress in the 'My Submissions' section
    5. You will receive email notifications at each stage
    """)

def show_performance_preview(submission_data, analyzer):
    """Show performance preview for basic form submission"""
    st.subheader("📊 Performance Preview")
    
    # Calculate estimated performance score
    estimated_score = estimate_performance_score(submission_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimated Performance Score", f"{estimated_score:.1f}/10")
    
    with col2:
        # Determine risk level
        if estimated_score >= 8.0:
            risk_level = "Low Risk"
            st.success(f"Risk Level: {risk_level}")
        elif estimated_score >= 6.0:
            risk_level = "Medium Risk"
            st.info(f"Risk Level: {risk_level}")
        else:
            risk_level = "High Risk"
            st.error(f"Risk Level: {risk_level}")
    
    with col3:
        # Determine approval recommendation
        if estimated_score >= 8.0:
            recommendation = "Full Approval - 5 Years"
        elif estimated_score >= 7.0:
            recommendation = "Provisional Approval - 3 Years"
        elif estimated_score >= 6.0:
            recommendation = "Conditional Approval - 1 Year"
        elif estimated_score >= 5.0:
            recommendation = "Approval with Strict Monitoring - 1 Year"
        else:
            recommendation = "Rejection - Significant Improvements Required"
        
        st.metric("Preliminary Recommendation", recommendation)

def estimate_performance_score(submission_data):
    """Estimate performance score from submitted data"""
    score = 5.0  # Base score
    
    academic_data = submission_data.get('academic_data', {})
    research_data = submission_data.get('research_data', {})
    infrastructure_data = submission_data.get('infrastructure_data', {})
    governance_data = submission_data.get('governance_data', {})
    
    # NAAC Grade
    naac_grade = academic_data.get('naac_grade', 'B')
    if naac_grade in ['A++', 'A+', 'A']:
        score += 1.5
    elif naac_grade in ['B++', 'B+']:
        score += 0.5
    
    # Placement Rate
    placement_rate = academic_data.get('placement_rate', 75)
    score += (placement_rate - 50) / 20  # Normalize
    
    # Student-Faculty Ratio
    sf_ratio = academic_data.get('student_faculty_ratio', 20)
    if sf_ratio <= 15:
        score += 1.0
    elif sf_ratio <= 20:
        score += 0.5
    
    # Research Publications
    research_pubs = research_data.get('research_publications', 0)
    score += min(1.5, research_pubs / 100)
    
    # Digital Infrastructure
    digital_score = infrastructure_data.get('digital_infrastructure_score', 7)
    score += (digital_score - 5) / 5
    
    # Financial Stability
    financial_score = governance_data.get('financial_stability_score', 8)
    score += (financial_score - 5) / 5
    
    return min(10.0, max(1.0, score))
