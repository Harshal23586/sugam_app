# institution/faculty.py
"""
Faculty Profile Management Module

This module handles faculty profile creation, document upload, and compliance tracking
for AICTE and UGC requirements.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import json
import os


def create_faculty_profile_tab(analyzer, user: Dict):
    """
    Create the faculty profile management interface
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: Dictionary containing user information
    """
    st.subheader("👨‍🏫 Faculty Profile Management")
    
    # Tabs for different faculty management functions
    faculty_tabs = st.tabs([
        "➕ Add New Faculty",
        "📋 View All Faculty",
        "📑 Faculty Document Checklist",
        "📊 Faculty Compliance Report"
    ])
    
    # Tab 1: Add New Faculty
    with faculty_tabs[0]:
        add_new_faculty_form(user)
    
    # Tab 2: View All Faculty
    with faculty_tabs[1]:
        view_all_faculty(user)
    
    # Tab 3: Faculty Document Checklist
    with faculty_tabs[2]:
        faculty_document_checklist()
    
    # Tab 4: Faculty Compliance Report
    with faculty_tabs[3]:
        faculty_compliance_report(analyzer, user)


def add_new_faculty_form(user: Dict):
    """Form to add new faculty member"""
    st.markdown("### Add New Faculty Member")
    
    with st.form("faculty_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            faculty_name = st.text_input("Full Name*")
            employee_id = st.text_input("Employee ID*")
            department = st.text_input("Department*")
            designation = st.selectbox(
                "Designation*",
                ["Professor", "Associate Professor", "Assistant Professor", "Lecturer", 
                 "Visiting Faculty", "Adjunct Professor", "Other"]
            )
            qualification = st.selectbox(
                "Highest Qualification*",
                ["PhD", "Post Doctoral", "M.Tech/M.E", "MBA", "MCA", "M.Sc", "B.Tech/B.E", "Other"]
            )
        
        with col2:
            email = st.text_input("Email ID*")
            phone = st.text_input("Phone Number")
            date_of_joining = st.date_input("Date of Joining*")
            employment_type = st.selectbox(
                "Employment Type*",
                ["Regular/Full-time", "Contract", "Visiting", "Adjunct", "Part-time"]
            )
            pan_number = st.text_input("PAN Number")
        
        st.markdown("---")
        st.markdown("#### Required Documents (AICTE & UGC Compliance)")
        
        # Document upload section
        doc_col1, doc_col2 = st.columns(2)
        
        with doc_col1:
            st.markdown("**Core Documents**")
            appointment_letter = st.file_uploader("Appointment Letter*", type=['pdf', 'doc', 'docx'])
            qualification_cert = st.file_uploader("Qualification Certificates*", type=['pdf', 'jpg', 'png'])
            experience_cert = st.file_uploader("Experience Certificates", type=['pdf', 'doc', 'docx'])
            pan_card = st.file_uploader("PAN Card*", type=['pdf', 'jpg', 'png'])
            aadhaar_card = st.file_uploader("Aadhaar Card*", type=['pdf', 'jpg', 'png'])
            photograph = st.file_uploader("Photograph*", type=['jpg', 'png', 'jpeg'])
        
        with doc_col2:
            st.markdown("**Compliance Documents**")
            salary_proof = st.file_uploader("Salary Proof (Last 6 months)*", type=['pdf'])
            form16 = st.file_uploader("Form-16/Income Tax Proof", type=['pdf'])
            epf_records = st.file_uploader("Aadhaar-Linked EPF Records", type=['pdf'])
            net_set_phd = st.file_uploader("NET/SET/PhD Certificate", type=['pdf', 'jpg', 'png'])
            publication_proof = st.file_uploader("Research Publications", type=['pdf'])
            joining_report = st.file_uploader("Joining Report", type=['pdf', 'doc', 'docx'])
        
        st.markdown("---")
        
        # Additional information
        st.markdown("#### Additional Information")
        
        col3, col4 = st.columns(2)
        
        with col3:
            net_set_status = st.selectbox(
                "NET/SET Status",
                ["NET Qualified", "SET Qualified", "PhD", "Not Applicable", "Not Qualified"]
            )
            pay_scale = st.text_input("UGC Pay Scale")
            biometric_enrolled = st.checkbox("Biometric Attendance Enrolled")
            fdp_attended = st.number_input("FDPs Attended", min_value=0, value=0)
        
        with col4:
            research_publications = st.number_input("Research Publications", min_value=0, value=0)
            patents = st.number_input("Patents Filed/Granted", min_value=0, value=0)
            projects_guided = st.number_input("Projects Guided", min_value=0, value=0)
            cas_applied = st.checkbox("CAS Applied")
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("➕ Add Faculty", use_container_width=True)
        
        if submitted:
            if not all([faculty_name, employee_id, department, designation, qualification, email]):
                st.error("Please fill all required fields (*)")
            else:
                # Save faculty data
                faculty_data = {
                    "faculty_id": f"FAC_{employee_id}_{datetime.now().strftime('%Y%m%d')}",
                    "employee_id": employee_id,
                    "name": faculty_name,
                    "department": department,
                    "designation": designation,
                    "qualification": qualification,
                    "email": email,
                    "phone": phone,
                    "date_of_joining": date_of_joining.isoformat() if date_of_joining else None,
                    "employment_type": employment_type,
                    "pan_number": pan_number,
                    "net_set_status": net_set_status,
                    "pay_scale": pay_scale,
                    "biometric_enrolled": biometric_enrolled,
                    "fdp_attended": fdp_attended,
                    "research_publications": research_publications,
                    "patents": patents,
                    "projects_guided": projects_guided,
                    "cas_applied": cas_applied,
                    "added_date": datetime.now().isoformat(),
                    "institution_id": user.get('institution_id'),
                    "documents": {
                        "appointment_letter": appointment_letter is not None,
                        "qualification_cert": qualification_cert is not None,
                        "experience_cert": experience_cert is not None,
                        "pan_card": pan_card is not None,
                        "aadhaar_card": aadhaar_card is not None,
                        "photograph": photograph is not None,
                        "salary_proof": salary_proof is not None,
                        "form16": form16 is not None,
                        "epf_records": epf_records is not None,
                        "net_set_phd": net_set_phd is not None,
                        "publication_proof": publication_proof is not None,
                        "joining_report": joining_report is not None
                    }
                }
                
                # Save to session state or database
                if 'faculty_data' not in st.session_state:
                    st.session_state.faculty_data = []
                
                st.session_state.faculty_data.append(faculty_data)
                
                # Save to file (for demo)
                save_faculty_data(faculty_data, user)
                
                st.success(f"✅ Faculty '{faculty_name}' added successfully!")
                st.info(f"Faculty ID: {faculty_data['faculty_id']}")


def view_all_faculty(user: Dict):
    """Display all faculty members in a table"""
    st.markdown("### Faculty Directory")
    
    # Load faculty data
    faculty_data = load_faculty_data(user)
    
    if not faculty_data:
        st.info("No faculty members added yet. Use 'Add New Faculty' tab to add faculty.")
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(faculty_data)
    
    # Select columns to display
    display_columns = ['name', 'employee_id', 'department', 'designation', 
                      'qualification', 'email', 'employment_type']
    
    if all(col in df.columns for col in display_columns):
        display_df = df[display_columns]
        
        # Search and filter
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search faculty by name, department, or designation")
        
        with col2:
            filter_dept = st.selectbox("Filter by Department", 
                                      ["All"] + list(df['department'].unique()))
        
        # Apply filters
        if search_term:
            mask = display_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            display_df = display_df[mask]
        
        if filter_dept != "All":
            display_df = display_df[display_df['department'] == filter_dept]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
        # Show statistics
        st.markdown("#### 📊 Faculty Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Faculty", len(df))
        
        with col2:
            regular_count = sum(1 for f in faculty_data if f.get('employment_type') == 'Regular/Full-time')
            st.metric("Regular Faculty", regular_count)
        
        with col3:
            phd_count = sum(1 for f in faculty_data if 'PhD' in str(f.get('qualification')))
            st.metric("PhD Holders", phd_count)
        
        with col4:
            avg_experience = calculate_average_experience(faculty_data)
            st.metric("Avg Experience", f"{avg_experience:.1f} years")
        
        # Detailed view option
        st.markdown("#### 👤 Faculty Details")
        if not display_df.empty:
            selected_faculty = st.selectbox(
                "Select faculty for detailed view",
                display_df['name'].tolist()
            )
            
            if selected_faculty:
                faculty = next((f for f in faculty_data if f['name'] == selected_faculty), None)
                if faculty:
                    show_faculty_details(faculty)


def faculty_document_checklist():
    """Display AICTE and UGC document requirements checklist"""
    st.markdown("### 📋 Faculty Document Checklist (AICTE & UGC)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **AICTE Requirements**")
        
        aicte_docs = [
            ("Appointment Letters", "Mandatory"),
            ("Joining Reports", "Mandatory"),
            ("Qualification Certificates (UG/PG/PhD)", "Mandatory"),
            ("AICTE Approved University Proof", "Mandatory"),
            ("Experience Certificates", "Mandatory"),
            ("PAN Card & Aadhaar", "Mandatory"),
            ("Faculty Photographs", "Mandatory"),
            ("Salary Proof (Bank statements, Salary slips)", "Mandatory"),
            ("Form-16 / Income Tax Proof", "Mandatory"),
            ("Faculty Undertaking (AICTE format)", "Mandatory"),
            ("Biometric Attendance Records", "Conditional"),
            ("Aadhaar-Linked EPF Records", "Conditional"),
            ("Faculty-Student Ratio Statement", "Mandatory"),
            ("Faculty Stability Certificate", "Mandatory"),
            ("Faculty Research Publications", "If applicable")
        ]
        
        for doc, status in aicte_docs:
            if "Mandatory" in status:
                st.checkbox(f"✅ {doc}", value=False, disabled=True)
            else:
                st.checkbox(f"📝 {doc}", value=False, disabled=True)
    
    with col2:
        st.markdown("#### **UGC Requirements**")
        
        ugc_docs = [
            ("Complete Faculty List (Department-wise)", "Mandatory"),
            ("Appointment & Selection Committee Minutes", "Mandatory"),
            ("Faculty Qualification Proof (UGC Norms)", "Mandatory"),
            ("NET / SET / PhD Certificates", "As applicable"),
            ("Salary Structure as per UGC Pay Scale", "Mandatory"),
            ("Proof of Full-Time Employment", "Mandatory"),
            ("Workload Distribution Register", "Mandatory"),
            ("Service Books of Faculty", "Mandatory"),
            ("Faculty Promotion Records (CAS)", "Mandatory"),
            ("Research Publications & Projects", "If applicable"),
            ("Faculty Attendance Register", "Mandatory"),
            ("University Approval of Faculty Appointments", "Mandatory"),
            ("Reservation Roster Compliance", "Mandatory"),
            ("Faculty Development Program (FDP) Records", "Mandatory")
        ]
        
        for doc, status in ugc_docs:
            if "Mandatory" in status:
                st.checkbox(f"✅ {doc}", value=False, disabled=True)
            else:
                st.checkbox(f"📝 {doc}", value=False, disabled=True)
    
    # Compliance status
    st.markdown("---")
    st.markdown("#### 📊 Document Compliance Status")
    
    # Calculate compliance (mock data for demo)
    compliance_data = {
        "AICTE Compliance": "75%",
        "UGC Compliance": "68%",
        "Total Documents Uploaded": "142/200",
        "Pending Mandatory": "8 documents"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AICTE Compliance", compliance_data["AICTE Compliance"])
    
    with col2:
        st.metric("UGC Compliance", compliance_data["UGC Compliance"])
    
    with col3:
        st.metric("Documents Uploaded", compliance_data["Total Documents Uploaded"])
    
    with col4:
        st.metric("Pending Mandatory", compliance_data["Pending Mandatory"])
    
    # Generate compliance report
    if st.button("📄 Generate Compliance Report", use_container_width=True):
        generate_compliance_report()


def faculty_compliance_report(analyzer, user: Dict):
    """Generate faculty compliance report and analytics"""
    st.markdown("### 📊 Faculty Compliance Report")
    
    # Load faculty data
    faculty_data = load_faculty_data(user)
    
    if not faculty_data:
        st.info("No faculty data available. Add faculty members first.")
        return
    
    # Calculate compliance metrics
    total_faculty = len(faculty_data)
    
    # Qualification analysis
    qualification_counts = {}
    for faculty in faculty_data:
        qual = faculty.get('qualification', 'Unknown')
        qualification_counts[qual] = qualification_counts.get(qual, 0) + 1
    
    # Employment type analysis
    employment_counts = {}
    for faculty in faculty_data:
        emp_type = faculty.get('employment_type', 'Unknown')
        employment_counts[emp_type] = employment_counts.get(emp_type, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Faculty", total_faculty)
    
    with col2:
        net_set_count = sum(1 for f in faculty_data if "Qualified" in str(f.get('net_set_status')))
        st.metric("NET/SET Qualified", net_set_count)
    
    with col3:
        phd_count = sum(1 for f in faculty_data if 'PhD' in str(f.get('qualification')))
        st.metric("PhD Holders", phd_count)
    
    with col4:
        regular_count = sum(1 for f in faculty_data if f.get('employment_type') == 'Regular/Full-time')
        st.metric("Regular Faculty", f"{regular_count} ({regular_count/total_faculty*100:.1f}%)")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Qualification Distribution")
        if qualification_counts:
            qual_df = pd.DataFrame(list(qualification_counts.items()), 
                                  columns=['Qualification', 'Count'])
            st.bar_chart(qual_df.set_index('Qualification'))
    
    with col2:
        st.markdown("#### 👥 Employment Type Distribution")
        if employment_counts:
            emp_df = pd.DataFrame(list(employment_counts.items()), 
                                 columns=['Employment Type', 'Count'])
            st.bar_chart(emp_df.set_index('Employment Type'))
    
    # Department-wise analysis
    st.markdown("#### 🏫 Department-wise Faculty Distribution")
    
    department_counts = {}
    for faculty in faculty_data:
        dept = faculty.get('department', 'Unknown')
        department_counts[dept] = department_counts.get(dept, 0) + 1
    
    if department_counts:
        dept_df = pd.DataFrame(list(department_counts.items()), 
                              columns=['Department', 'Faculty Count'])
        st.dataframe(dept_df.sort_values('Faculty Count', ascending=False), 
                    use_container_width=True)
    
    # Document compliance summary
    st.markdown("#### 📑 Document Compliance Summary")
    
    if faculty_data and 'documents' in faculty_data[0]:
        doc_columns = list(faculty_data[0]['documents'].keys())
        doc_compliance = {}
        
        for doc in doc_columns:
            compliant_count = sum(1 for f in faculty_data 
                                if f.get('documents', {}).get(doc, False))
            doc_compliance[doc] = compliant_count
        
        compliance_df = pd.DataFrame(list(doc_compliance.items()), 
                                    columns=['Document', 'Compliant Count'])
        compliance_df['Percentage'] = (compliance_df['Compliant Count'] / total_faculty * 100).round(1)
        
        st.dataframe(compliance_df, use_container_width=True)
    
    # Download report
    if st.button("📥 Download Faculty Compliance Report", use_container_width=True):
        download_faculty_report(faculty_data, user)


def show_faculty_details(faculty: Dict):
    """Display detailed information about a faculty member"""
    with st.expander(f"👤 Details: {faculty.get('name')}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Employee ID:** {faculty.get('employee_id', 'N/A')}")
            st.markdown(f"**Department:** {faculty.get('department', 'N/A')}")
            st.markdown(f"**Designation:** {faculty.get('designation', 'N/A')}")
            st.markdown(f"**Qualification:** {faculty.get('qualification', 'N/A')}")
            st.markdown(f"**NET/SET Status:** {faculty.get('net_set_status', 'N/A')}")
            st.markdown(f"**Date of Joining:** {faculty.get('date_of_joining', 'N/A')}")
        
        with col2:
            st.markdown(f"**Email:** {faculty.get('email', 'N/A')}")
            st.markdown(f"**Phone:** {faculty.get('phone', 'N/A')}")
            st.markdown(f"**Employment Type:** {faculty.get('employment_type', 'N/A')}")
            st.markdown(f"**UGC Pay Scale:** {faculty.get('pay_scale', 'N/A')}")
            st.markdown(f"**Research Publications:** {faculty.get('research_publications', 0)}")
            st.markdown(f"**FDPs Attended:** {faculty.get('fdp_attended', 0)}")
        
        # Documents status
        st.markdown("#### 📄 Document Status")
        
        if 'documents' in faculty:
            doc_col1, doc_col2 = st.columns(2)
            
            uploaded_docs = [doc for doc, status in faculty['documents'].items() if status]
            pending_docs = [doc for doc, status in faculty['documents'].items() if not status]
            
            with doc_col1:
                st.markdown("**✅ Uploaded:**")
                for doc in uploaded_docs:
                    st.markdown(f"✓ {doc.replace('_', ' ').title()}")
            
            with doc_col2:
                st.markdown("**📝 Pending:**")
                for doc in pending_docs:
                    st.markdown(f"⚠ {doc.replace('_', ' ').title()}")


def save_faculty_data(faculty_data: Dict, user: Dict):
    """Save faculty data to file (for demo purposes)"""
    try:
        filename = f"faculty_data_{user.get('institution_id')}.json"
        
        # Load existing data
        existing_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        
        # Add new data
        existing_data.append(faculty_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving faculty data: {str(e)}")


def load_faculty_data(user: Dict) -> List[Dict]:
    """Load faculty data from file or session state"""
    try:
        # Try to load from session state first
        if 'faculty_data' in st.session_state and st.session_state.faculty_data:
            return st.session_state.faculty_data
        
        # Try to load from file
        filename = f"faculty_data_{user.get('institution_id')}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading faculty data: {str(e)}")
    
    return []


def calculate_average_experience(faculty_data: List[Dict]) -> float:
    """Calculate average experience of faculty (mock implementation)"""
    # This is a simplified version - in real implementation, 
    # you would calculate based on date_of_joining
    return 5.5  # Mock average


def generate_compliance_report():
    """Generate and display compliance report"""
    st.success("Compliance report generated successfully!")
    
    report_data = {
        "Generated Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AICTE Compliance Score": "75%",
        "UGC Compliance Score": "68%",
        "Total Mandatory Documents": "28",
        "Documents Completed": "20",
        "Documents Pending": "8",
        "Critical Pending": "3",
        "Estimated Time to Completion": "2 weeks"
    }
    
    st.markdown("#### 📊 Compliance Summary")
    for key, value in report_data.items():
        st.markdown(f"**{key}:** {value}")
    
    # Download option
    report_json = json.dumps(report_data, indent=2)
    st.download_button(
        label="📥 Download Compliance Report",
        data=report_json,
        file_name=f"faculty_compliance_report_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )


def download_faculty_report(faculty_data: List[Dict], user: Dict):
    """Generate and download faculty report"""
    report = {
        "report_type": "Faculty Compliance Report",
        "generated_date": datetime.now().isoformat(),
        "institution": user.get('institution_name'),
        "institution_id": user.get('institution_id'),
        "total_faculty": len(faculty_data),
        "summary": {
            "phd_holders": sum(1 for f in faculty_data if 'PhD' in str(f.get('qualification'))),
            "net_set_qualified": sum(1 for f in faculty_data if "Qualified" in str(f.get('net_set_status'))),
            "regular_faculty": sum(1 for f in faculty_data if f.get('employment_type') == 'Regular/Full-time'),
            "average_experience": calculate_average_experience(faculty_data)
        },
        "department_distribution": {},
        "document_compliance": {}
    }
    
    # Calculate department distribution
    for faculty in faculty_data:
        dept = faculty.get('department', 'Unknown')
        report["department_distribution"][dept] = report["department_distribution"].get(dept, 0) + 1
    
    # Calculate document compliance
    if faculty_data and 'documents' in faculty_data[0]:
        doc_columns = list(faculty_data[0]['documents'].keys())
        for doc in doc_columns:
            compliant_count = sum(1 for f in faculty_data 
                                if f.get('documents', {}).get(doc, False))
            report["document_compliance"][doc] = {
                "compliant": compliant_count,
                "percentage": (compliant_count / len(faculty_data) * 100) if faculty_data else 0
            }
    
    # Convert to JSON for download
    report_json = json.dumps(report, indent=2)
    
    st.download_button(
        label="📥 Download Faculty Report (JSON)",
        data=report_json,
        file_name=f"faculty_report_{user.get('institution_id')}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )
