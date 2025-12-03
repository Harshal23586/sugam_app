import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json

def create_institution_document_upload(analyzer, user):
    st.subheader("📤 Document Upload Portal")
    
    st.info("""
    **Upload and manage all required documents for approval processes.**
    This portal supports parameter-wise document submission based on NEP 2020 framework.
    """)
    
    # Two-step process: Parameter selection then document upload
    tab1, tab2, tab3 = st.tabs([
        "📋 Document Requirements", 
        "📤 Upload Documents", 
        "📊 Upload Analysis"
    ])
    
    with tab1:
        show_document_requirements(analyzer)
    
    with tab2:
        upload_documents_section(analyzer, user)
    
    with tab3:
        show_upload_analysis(analyzer, user)

def show_document_requirements(analyzer):
    """Show document requirements organized by parameters"""
    st.subheader("📋 Document Requirements by Parameters")
    
    approval_type = st.selectbox(
        "Select Approval Type",
        ["New Approval", "Renewal Approval", "Expansion Approval", "Accreditation"],
        key="requirements_type"
    )
    
    # Document requirements organized by 10 parameters
    requirements = get_document_requirements_by_parameters(approval_type)
    
    # Show parameter-wise requirements
    for param_num, (parameter, docs) in enumerate(requirements.items(), 1):
        with st.expander(f"📚 {parameter.upper()}", expanded=(param_num == 1)):
            st.markdown(f"#### Mandatory Documents")
            
            for i, doc in enumerate(docs['mandatory'], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{i}. {doc}")
                with col2:
                    st.info("**Required**")
                with col3:
                    st.write("📄 PDF/DOCX")
            
            if docs.get('supporting'):
                st.markdown(f"#### Supporting Documents")
                for i, doc in enumerate(docs['supporting'], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i}. {doc}")
                    with col2:
                        st.info("**Optional**")
    
    # Download requirements checklist
    st.markdown("---")
    st.subheader("📥 Download Checklist")
    
    checklist_text = generate_checklist_text(requirements, approval_type)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📋 Download Requirements Checklist",
            data=checklist_text,
            file_name=f"document_requirements_{approval_type.replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    with col2:
        st.download_button(
            label="📋 Download Excel Template",
            data=generate_excel_template(requirements),
            file_name=f"document_tracker_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def upload_documents_section(analyzer, user):
    """Main document upload section"""
    st.subheader("📤 Upload Documents")
    
    # Step 1: Select parameter
    st.markdown("### Step 1: Select Parameter")
    
    parameters = [
        "1. Curriculum",
        "2. Faculty Resources",
        "3. Learning and Teaching",
        "4. Research and Innovation",
        "5. Extracurricular & Co-curricular Activities",
        "6. Community Engagement",
        "7. Green Initiatives",
        "8. Governance and Administration",
        "9. Infrastructure Development",
        "10. Financial Resources and Management"
    ]
    
    selected_parameter = st.selectbox(
        "Select Parameter for Document Upload",
        parameters,
        key="selected_parameter"
    )
    
    # Step 2: Select document type
    st.markdown("### Step 2: Select Document Type")
    
    document_categories = {
        "1. Curriculum": {
            "Mandatory": [
                "Curriculum framework and syllabus documents",
                "Course outlines with learning objectives",
                "Evidence of curriculum review processes",
                "Academic calendar and schedules",
                "Student feedback reports on curriculum"
            ],
            "Supporting": [
                "Innovative teaching-learning materials",
                "Industry interface documents",
                "Multidisciplinary course documentation",
                "Skill-integration evidence",
                "Vocational course records"
            ]
        },
        "2. Faculty Resources": {
            "Mandatory": [
                "Faculty qualification records",
                "Faculty recruitment policy",
                "Selection committee minutes",
                "Faculty development programs records",
                "Faculty appraisal records"
            ],
            "Supporting": [
                "Faculty achievement records",
                "Research publication records",
                "Industry exposure evidence",
                "Conference participation records",
                "Professional development plans"
            ]
        },
        "3. Learning and Teaching": {
            "Mandatory": [
                "Teaching plans and schedules",
                "Student assessment records",
                "Learning outcome achievement records",
                "Classroom observation reports",
                "Digital learning infrastructure details"
            ],
            "Supporting": [
                "Innovative teaching methodology docs",
                "Experiential learning records",
                "Student project documentation",
                "Research-oriented learning evidence",
                "Critical thinking activities"
            ]
        },
        "4. Research and Innovation": {
            "Mandatory": [
                "Research policy document",
                "Research publication records",
                "Patent filings documentation",
                "Research project funding details",
                "Research collaboration agreements"
            ],
            "Supporting": [
                "Research facility details",
                "Seminar/conference organization records",
                "Industry research partnerships",
                "Student research participation",
                "Translational research outcomes"
            ]
        },
        "5. Extracurricular & Co-curricular Activities": {
            "Mandatory": [
                "EC/CC activity calendar",
                "Student participation records",
                "Activity reports and outcomes",
                "Credit allocation policy",
                "Club/society registration"
            ],
            "Supporting": [
                "Awards and achievements",
                "Leadership program records",
                "Community service reports",
                "Student representation records",
                "Event photographs/videos"
            ]
        },
        "6. Community Engagement": {
            "Mandatory": [
                "Community engagement policy",
                "Outreach program records",
                "Social project documentation",
                "Village adoption records",
                "Student internship reports"
            ],
            "Supporting": [
                "Community feedback reports",
                "Collaborative project agreements",
                "Social research publications",
                "CSR initiative documentation",
                "Public awareness campaign records"
            ]
        },
        "7. Green Initiatives": {
            "Mandatory": [
                "Environmental policy document",
                "Energy consumption records",
                "Waste management system docs",
                "Water harvesting records",
                "Green building certification"
            ],
            "Supporting": [
                "Renewable energy installation",
                "Environmental audit reports",
                "Sustainability project docs",
                "Green campus initiatives",
                "Environmental awareness programs"
            ]
        },
        "8. Governance and Administration": {
            "Mandatory": [
                "Institutional act and statutes",
                "Organizational structure chart",
                "Governance body minutes",
                "Financial management policies",
                "Grievance redressal records"
            ],
            "Supporting": [
                "e-Governance implementation",
                "Strategic plans and reports",
                "International collaboration agreements",
                "Stakeholder satisfaction surveys",
                "Decision-making process docs"
            ]
        },
        "9. Infrastructure Development": {
            "Mandatory": [
                "Campus master plan",
                "Building and facility inventory",
                "Laboratory equipment details",
                "Library resource documentation",
                "IT infrastructure details"
            ],
            "Supporting": [
                "Infrastructure utilization reports",
                "Maintenance and upgrade records",
                "Safety and security system docs",
                "Accessibility compliance",
                "Future development plans"
            ]
        },
        "10. Financial Resources and Management": {
            "Mandatory": [
                "Annual financial statements",
                "Budget allocation certificates",
                "Salary expenditure records",
                "Research grant utilization",
                "Infrastructure expenditure"
            ],
            "Supporting": [
                "Financial planning documents",
                "Resource mobilization records",
                "Academic development investment",
                "Student scholarship details",
                "Revenue generation analysis"
            ]
        }
    }
    
    # Get documents for selected parameter
    param_docs = document_categories.get(selected_parameter, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc_category = st.radio(
            "Document Category",
            ["Mandatory", "Supporting"],
            horizontal=True,
            key="doc_category"
        )
    
    with col2:
        # Auto-suggest document names based on category
        if doc_category == "Mandatory":
            doc_options = param_docs.get("Mandatory", [])
        else:
            doc_options = param_docs.get("Supporting", [])
        
        if doc_options:
            suggested_doc = st.selectbox(
                "Select Document Type",
                doc_options,
                key="suggested_doc"
            )
        else:
            suggested_doc = st.text_input(
                "Document Name",
                placeholder="Enter document name...",
                key="custom_doc"
            )
    
    # Step 3: Upload documents
    st.markdown("### Step 3: Upload Document")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png', 'jpeg'],
        accept_multiple_files=True,
        help="Maximum file size: 10MB per file",
        key="doc_uploader"
    )
    
    if uploaded_files:
        # File validation and preview
        st.markdown("#### 📄 File Preview")
        
        for i, file in enumerate(uploaded_files):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{file.name}**")
                st.caption(f"Size: {file.size / 1024:.1f} KB | Type: {file.type}")
            
            with col2:
                # File type icon
                if file.type == 'application/pdf':
                    st.write("📄 PDF")
                elif 'word' in file.type:
                    st.write("📝 DOC")
                elif 'excel' in file.type:
                    st.write("📊 XLSX")
                elif 'image' in file.type:
                    st.write("🖼️ IMG")
                else:
                    st.write("📁 FILE")
            
            with col3:
                # File status
                st.success("✅ Ready")
            
            with col4:
                # Remove file button
                if st.button("🗑️", key=f"remove_{i}"):
                    uploaded_files.pop(i)
                    st.rerun()
        
        # Additional metadata
        st.markdown("#### 📝 Document Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            document_version = st.text_input(
                "Document Version",
                value="1.0",
                help="Version number of the document"
            )
            
            effective_date = st.date_input(
                "Effective Date",
                value=datetime.now().date(),
                help="When the document becomes effective"
            )
        
        with col2:
            confidentiality = st.selectbox(
                "Confidentiality Level",
                ["Public", "Internal", "Confidential", "Restricted"],
                help="Document access restrictions"
            )
            
            retention_period = st.selectbox(
                "Retention Period",
                ["1 Year", "3 Years", "5 Years", "10 Years", "Permanent"],
                help="How long to retain the document"
            )
        
        # Submission notes
        submission_notes = st.text_area(
            "Additional Notes",
            placeholder="Add any notes about these documents...",
            height=100
        )
        
        # Submit button
        if st.button("🚀 Upload Documents", type="primary"):
            if upload_documents(analyzer, user, uploaded_files, selected_parameter, 
                              suggested_doc, document_version, effective_date, 
                              confidentiality, retention_period, submission_notes):
                st.success("✅ Documents uploaded successfully!")
                st.balloons()
                
                # Show next steps
                st.info("""
                **Next Steps:**
                1. Documents are now in the verification queue
                2. AI analysis will begin automatically
                3. Check 'Upload Analysis' tab for status
                4. You will receive email notifications
                """)
                
                # Clear the form
                st.session_state.doc_uploader = []
                st.rerun()

def show_upload_analysis(analyzer, user):
    """Show analysis of uploaded documents"""
    st.subheader("📊 Document Upload Analysis")
    
    # Get uploaded documents for this institution
    try:
        uploaded_docs = analyzer.get_institution_documents(user['institution_id'])
        
        if uploaded_docs.empty:
            st.info("No documents uploaded yet. Use the 'Upload Documents' tab to get started.")
            return
        
        # Overall statistics
        st.markdown("### 📈 Upload Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = len(uploaded_docs)
            st.metric("Total Documents", total_docs)
        
        with col2:
            uploaded_today = len(uploaded_docs[
                pd.to_datetime(uploaded_docs['upload_date']).dt.date == datetime.now().date()
            ])
            st.metric("Uploaded Today", uploaded_today)
        
        with col3:
            pending_docs = len(uploaded_docs[uploaded_docs['status'] == 'Pending'])
            st.metric("Pending Review", pending_docs)
        
        with col4:
            approved_docs = len(uploaded_docs[uploaded_docs['status'] == 'Approved'])
            st.metric("Approved", approved_docs)
        
        # Document status visualization
        st.markdown("### 📊 Document Status Overview")
        
        status_counts = uploaded_docs['status'].value_counts()
        
        fig1 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Document Status Distribution",
            color=status_counts.index,
            color_discrete_map={
                'Approved': '#2ecc71',
                'Pending': '#f39c12',
                'Rejected': '#e74c3c',
                'Under Review': '#3498db'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Parameter-wise coverage
        st.markdown("### 📚 Parameter-wise Coverage")
        
        # Extract parameter from document names/types
        parameter_coverage = {}
        for _, doc in uploaded_docs.iterrows():
            doc_name = doc['document_name'].lower()
            doc_type = doc['document_type'].lower()
            
            # Map to parameters
            for param in [
                "curriculum", "faculty", "learning", "research", 
                "extracurricular", "community", "green", "governance",
                "infrastructure", "financial"
            ]:
                if param in doc_name or param in doc_type:
                    parameter_coverage[param] = parameter_coverage.get(param, 0) + 1
        
        if parameter_coverage:
            df_coverage = pd.DataFrame({
                'Parameter': [p.title() for p in parameter_coverage.keys()],
                'Documents': list(parameter_coverage.values())
            }).sort_values('Documents', ascending=False)
            
            fig2 = px.bar(
                df_coverage,
                x='Parameter',
                y='Documents',
                title="Documents by Parameter",
                color='Documents',
                color_continuous_scale='Viridis'
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recent uploads table
        st.markdown("### 📋 Recent Uploads")
        
        # Format the dataframe for display
        display_cols = ['document_name', 'document_type', 'upload_date', 'status']
        recent_docs = uploaded_docs[display_cols].sort_values('upload_date', ascending=False).head(10)
        
        # Apply styling
        def color_status(val):
            if val == 'Approved':
                return 'color: green; font-weight: bold'
            elif val == 'Rejected':
                return 'color: red; font-weight: bold'
            elif val == 'Pending':
                return 'color: orange; font-weight: bold'
            else:
                return 'color: blue; font-weight: bold'
        
        styled_df = recent_docs.style.applymap(color_status, subset=['status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # AI Analysis Recommendations
        st.markdown("### 🤖 AI Analysis & Recommendations")
        
        if st.button("🔍 Run Document Analysis", type="secondary"):
            with st.spinner("Analyzing documents..."):
                analysis_results = analyze_documents_ai(uploaded_docs)
                show_ai_analysis(analysis_results)
        
        # Document checklist progress
        st.markdown("### 📋 Document Checklist Progress")
        
        # Simulated requirements checklist
        requirements = get_all_document_requirements()
        uploaded_doc_names = uploaded_docs['document_name'].str.lower().tolist()
        
        progress_data = []
        for param, docs in requirements.items():
            total_docs = len(docs['mandatory'])
            uploaded_count = 0
            
            for doc in docs['mandatory']:
                if any(doc.lower() in uploaded_doc for uploaded_doc in uploaded_doc_names):
                    uploaded_count += 1
            
            progress = (uploaded_count / total_docs * 100) if total_docs > 0 else 0
            progress_data.append({
                'Parameter': param,
                'Progress': progress,
                'Uploaded': uploaded_count,
                'Total': total_docs
            })
        
        df_progress = pd.DataFrame(progress_data)
        
        # Create progress bars
        for _, row in df_progress.iterrows():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{row['Parameter']}**")
            with col2:
                st.progress(row['Progress'] / 100)
            with col3:
                st.write(f"{row['Uploaded']}/{row['Total']}")
        
        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = uploaded_docs.to_csv(index=False)
            st.download_button(
                label="📊 Export to CSV",
                data=csv_data,
                file_name=f"documents_{user['institution_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate report
            report_text = generate_document_report(uploaded_docs, user)
            st.download_button(
                label="📋 Download Report",
                data=report_text,
                file_name=f"document_report_{user['institution_id']}.txt",
                mime="text/plain"
            )
        
        with col3:
            if st.button("🔄 Refresh Analysis"):
                st.rerun()
        
    except Exception as e:
        st.error(f"Error loading document data: {str(e)}")

def upload_documents(analyzer, user, uploaded_files, parameter, document_type, 
                    version, effective_date, confidentiality, retention_period, notes):
    """Handle document upload and storage"""
    try:
        for file in uploaded_files:
            # Save document metadata to database
            cursor = analyzer.conn.cursor()
            
            document_data = {
                'institution_id': user['institution_id'],
                'document_name': file.name,
                'document_type': f"{parameter} - {document_type}",
                'parameter': parameter.split('. ')[1] if '. ' in parameter else parameter,
                'category': st.session_state.get('doc_category', 'Mandatory'),
                'version': version,
                'effective_date': effective_date.isoformat(),
                'confidentiality': confidentiality,
                'retention_period': retention_period,
                'uploaded_by': user['contact_person'],
                'upload_date': datetime.now().isoformat(),
                'file_size_kb': file.size / 1024,
                'file_type': file.type,
                'status': 'Pending',
                'notes': notes,
                'ai_analysis': 'Pending'
            }
            
            cursor.execute('''
                INSERT INTO institution_documents 
                (institution_id, document_name, document_type, parameter, category,
                 version, effective_date, confidentiality, retention_period,
                 uploaded_by, upload_date, file_size_kb, file_type, status, notes, ai_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_data['institution_id'],
                document_data['document_name'],
                document_data['document_type'],
                document_data['parameter'],
                document_data['category'],
                document_data['version'],
                document_data['effective_date'],
                document_data['confidentiality'],
                document_data['retention_period'],
                document_data['uploaded_by'],
                document_data['upload_date'],
                document_data['file_size_kb'],
                document_data['file_type'],
                document_data['status'],
                document_data['notes'],
                document_data['ai_analysis']
            ))
        
        analyzer.conn.commit()
        
        # Trigger AI analysis in background
        trigger_ai_analysis(analyzer, user['institution_id'])
        
        return True
        
    except Exception as e:
        st.error(f"Error uploading documents: {str(e)}")
        return False

def analyze_documents_ai(uploaded_docs):
    """Simulate AI document analysis"""
    analysis_results = {
        'total_documents': len(uploaded_docs),
        'analysis_date': datetime.now().isoformat(),
        'recommendations': [],
        'warnings': [],
        'completeness_score': 0,
        'quality_score': 0
    }
    
    # Simulate analysis based on document metadata
    mandatory_docs = uploaded_docs[uploaded_docs['category'] == 'Mandatory']
    supporting_docs = uploaded_docs[uploaded_docs['category'] == 'Supporting']
    
    analysis_results['mandatory_count'] = len(mandatory_docs)
    analysis_results['supporting_count'] = len(supporting_docs)
    
    # Calculate completeness score
    total_required = 50  # Simulated total required documents
    completeness = (len(uploaded_docs) / total_required) * 100
    analysis_results['completeness_score'] = min(100, completeness)
    
    # Generate recommendations
    if len(mandatory_docs) < 30:
        analysis_results['recommendations'].append(
            "Focus on uploading more mandatory documents for comprehensive assessment"
        )
    
    if len(supporting_docs) > 20:
        analysis_results['quality_score'] = 85
        analysis_results['recommendations'].append(
            "Excellent supporting documentation. Consider submitting for premium accreditation"
        )
    else:
        analysis_results['quality_score'] = 65
        analysis_results['recommendations'].append(
            "Add more supporting documents to strengthen your application"
        )
    
    # Check for common issues
    pdf_count = len(uploaded_docs[uploaded_docs['file_type'] == 'application/pdf'])
    if pdf_count < len(uploaded_docs) * 0.8:
        analysis_results['warnings'].append(
            "Consider converting documents to PDF format for better compatibility"
        )
    
    return analysis_results

def show_ai_analysis(analysis_results):
    """Display AI analysis results"""
    st.markdown("#### 🤖 AI Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Completeness Score", f"{analysis_results['completeness_score']:.0f}%")
    
    with col2:
        st.metric("Quality Score", f"{analysis_results['quality_score']:.0f}/100")
    
    with col3:
        st.metric("Documents Analyzed", analysis_results['total_documents'])
    
    # Recommendations
    if analysis_results['recommendations']:
        st.markdown("#### 💡 Recommendations")
        for rec in analysis_results['recommendations']:
            st.info(f"• {rec}")
    
    # Warnings
    if analysis_results['warnings']:
        st.markdown("#### ⚠️ Areas for Improvement")
        for warning in analysis_results['warnings']:
            st.warning(f"• {warning}")
    
    # Visual analysis
    st.markdown("#### 📊 Analysis Summary")
    
    scores_data = {
        'Metric': ['Completeness', 'Quality', 'Mandatory Docs', 'Supporting Docs'],
        'Score': [
            analysis_results['completeness_score'],
            analysis_results['quality_score'],
            analysis_results.get('mandatory_count', 0),
            analysis_results.get('supporting_count', 0)
        ]
    }
    
    df_scores = pd.DataFrame(scores_data)
    
    fig = px.bar(df_scores, x='Metric', y='Score', 
                title="Document Analysis Summary",
                color='Score',
                color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig, use_container_width=True)

def get_document_requirements_by_parameters(approval_type):
    """Get document requirements organized by parameters"""
    # This is a comprehensive list of document requirements
    # In a real application, this would come from a database or configuration
    
    base_requirements = {
        "1. Curriculum": {
            "mandatory": [
                "Curriculum framework documents for all programs",
                "Course outlines with learning objectives and outcomes",
                "Evidence of curriculum review and revision processes",
                "Academic calendar and course schedules",
                "Student feedback reports on curriculum"
            ],
            "supporting": [
                "Innovative teaching-learning materials developed",
                "Industry interface documents for curriculum design",
                "Multidisciplinary course documentation",
                "Skill-integration evidence in curriculum",
                "Vocational and skill-based course records"
            ]
        },
        "2. Faculty Resources": {
            "mandatory": [
                "Faculty qualification records and biodata",
                "Faculty recruitment policy and procedures",
                "Selection committee composition and minutes",
                "Faculty development programs records",
                "Faculty appraisal and performance records"
            ],
            "supporting": [
                "Faculty achievement and award records",
                "Research publication records",
                "Industry exposure evidence",
                "Conference participation records",
                "Professional development plans"
            ]
        },
        "3. Learning and Teaching": {
            "mandatory": [
                "Teaching plans and lesson schedules",
                "Student assessment records and evaluation methods",
                "Learning outcome achievement records",
                "Classroom observation reports",
                "Digital learning infrastructure details"
            ],
            "supporting": [
                "Innovative teaching methodology documentation",
                "Experiential learning activity records",
                "Student project documentation",
                "Research-oriented learning evidence",
                "Critical thinking development activities"
            ]
        },
        "4. Research and Innovation": {
            "mandatory": [
                "Research policy document",
                "Research publication records (citations, impact factors)",
                "Patent filings and grants documentation",
                "Research project funding details",
                "Research collaboration agreements"
            ],
            "supporting": [
                "Research facility details and utilization",
                "Research seminar/conference organization records",
                "Industry research partnership documents",
                "Student research participation records",
                "Translational research outcomes"
            ]
        },
        "5. Extracurricular Activities": {
            "mandatory": [
                "EC/CC activity calendar and schedules",
                "Student participation records",
                "Activity reports and outcomes",
                "Credit allocation policy for EC/CC activities",
                "Student club and society registration"
            ],
            "supporting": [
                "Awards and achievements in EC/CC activities",
                "Leadership development program records",
                "Community service activity reports",
                "Student representation in governance bodies",
                "Event photographs and videos"
            ]
        },
        "6. Community Engagement": {
            "mandatory": [
                "Community engagement policy",
                "Outreach program records and reports",
                "Social project documentation",
                "Village/community adoption records",
                "Student internship reports with community focus"
            ],
            "supporting": [
                "Community feedback and impact assessment",
                "Collaborative project agreements",
                "Social research publications",
                "CSR initiative documentation",
                "Public awareness campaign records"
            ]
        },
        "7. Green Initiatives": {
            "mandatory": [
                "Environmental policy document",
                "Energy consumption and conservation records",
                "Waste management system documentation",
                "Water harvesting and recycling records",
                "Green building certification (if any)"
            ],
            "supporting": [
                "Renewable energy installation details",
                "Environmental audit reports",
                "Sustainability project documentation",
                "Green campus initiative records",
                "Environmental awareness program reports"
            ]
        },
        "8. Governance and Administration": {
            "mandatory": [
                "Institutional act, statutes, and regulations",
                "Organizational structure chart",
                "Governance body composition and minutes",
                "Financial management policies",
                "Grievance redressal mechanism records"
            ],
            "supporting": [
                "e-Governance implementation details",
                "Strategic plans and implementation reports",
                "International collaboration agreements",
                "Stakeholder satisfaction surveys",
                "Decision-making process documentation"
            ]
        },
        "9. Infrastructure Development": {
            "mandatory": [
                "Campus master plan and layout",
                "Building and facility inventory",
                "Laboratory and equipment details",
                "Library resource documentation",
                "IT infrastructure details"
            ],
            "supporting": [
                "Infrastructure utilization reports",
                "Maintenance and upgrade records",
                "Safety and security system details",
                "Accessibility compliance documentation",
                "Future development plans"
            ]
        },
        "10. Financial Management": {
            "mandatory": [
                "Annual financial statements and audit reports",
                "Budget allocation and utilization certificates",
                "Salary expenditure records",
                "Research grant utilization details",
                "Infrastructure development expenditure"
            ],
            "supporting": [
                "Financial planning documents",
                "Resource mobilization records",
                "Investment in academic development",
                "Student scholarship and financial aid details",
                "Revenue generation from various sources"
            ]
        }
    }
    
    return base_requirements

def get_all_document_requirements():
    """Get all document requirements across parameters"""
    requirements = {}
    approval_types = ["New Approval", "Renewal Approval", "Expansion Approval"]
    
    for approval_type in approval_types:
        reqs = get_document_requirements_by_parameters(approval_type)
        for param, docs in reqs.items():
            if param not in requirements:
                requirements[param] = {'mandatory': set(), 'supporting': set()}
            
            requirements[param]['mandatory'].update(docs['mandatory'])
            requirements[param]['supporting'].update(docs.get('supporting', []))
    
    # Convert sets back to lists
    for param in requirements:
        requirements[param]['mandatory'] = list(requirements[param]['mandatory'])
        requirements[param]['supporting'] = list(requirements[param]['supporting'])
    
    return requirements

def generate_checklist_text(requirements, approval_type):
    """Generate checklist text for download"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"DOCUMENT REQUIREMENTS CHECKLIST")
    lines.append("=" * 60)
    lines.append(f"Approval Type: {approval_type}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    for param_num, (parameter, docs) in enumerate(requirements.items(), 1):
        lines.append(f"{parameter.upper()}")
        lines.append("-" * 40)
        
        lines.append("Mandatory Documents:")
        for i, doc in enumerate(docs['mandatory'], 1):
            lines.append(f"  {i}. [ ] {doc}")
        
        if docs.get('supporting'):
            lines.append("")
            lines.append("Supporting Documents:")
            for i, doc in enumerate(docs['supporting'], 1):
                lines.append(f"  {i}. [ ] {doc}")
        
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("Instructions:")
    lines.append("1. Check the box when document is uploaded")
    lines.append("2. Maintain digital copies of all documents")
    lines.append("3. Keep original documents for verification")
    lines.append("=" * 60)
    
    return "\n".join(lines)

def generate_excel_template(requirements):
    """Generate Excel template for document tracking"""
    # This would generate an Excel file using pandas ExcelWriter
    # For now, return a placeholder
    import io
    
    output = io.BytesIO()
    
    # Create DataFrame for each parameter
    dfs = []
    for param, docs in requirements.items():
        param_df = pd.DataFrame({
            'Parameter': [param] * (len(docs['mandatory']) + len(docs.get('supporting', []))),
            'Document Type': ['Mandatory'] * len(docs['mandatory']) + ['Supporting'] * len(docs.get('supporting', [])),
            'Document Name': docs['mandatory'] + docs.get('supporting', []),
            'Status': ['Pending'] * (len(docs['mandatory']) + len(docs.get('supporting', []))),
            'Upload Date': [''] * (len(docs['mandatory']) + len(docs.get('supporting', []))),
            'Remarks': [''] * (len(docs['mandatory']) + len(docs.get('supporting', [])))
        })
        dfs.append(param_df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Document Tracker', index=False)
            
            # Get workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets['Document Tracker']
            
            # Add some basic formatting
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.getvalue()

def generate_document_report(uploaded_docs, user):
    """Generate a comprehensive document report"""
    lines = []
    
    lines.append("=" * 60)
    lines.append(f"INSTITUTIONAL DOCUMENTS REPORT")
    lines.append("=" * 60)
    lines.append(f"Institution: {user.get('institution_name', 'N/A')}")
    lines.append(f"Institution ID: {user['institution_id']}")
    lines.append(f"Contact Person: {user['contact_person']}")
    lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Documents: {len(uploaded_docs)}")
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    
    status_counts = uploaded_docs['status'].value_counts()
    for status, count in status_counts.items():
        lines.append(f"{status}: {count}")
    
    lines.append("")
    
    # Document details
    lines.append("DOCUMENT DETAILS")
    lines.append("-" * 40)
    
    for _, doc in uploaded_docs.iterrows():
        lines.append(f"Document: {doc['document_name']}")
        lines.append(f"  Type: {doc.get('document_type', 'N/A')}")
        lines.append(f"  Parameter: {doc.get('parameter', 'N/A')}")
        lines.append(f"  Category: {doc.get('category', 'N/A')}")
        lines.append(f"  Status: {doc['status']}")
        lines.append(f"  Upload Date: {doc['upload_date']}")
        lines.append(f"  File Size: {doc.get('file_size_kb', 0):.1f} KB")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60
