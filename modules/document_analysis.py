import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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

def generate_enhanced_dummy_document_data(analyzer):
    """Generate realistic dummy document data with upload patterns and dates"""
    
    if 'enhanced_docs_generated' not in st.session_state:
        try:
            # Get all institutions (20 institutions)
            institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
            
            for institution_id in institutions:
                # Get institution performance to determine document completeness
                performance = get_institution_performance(institution_id, analyzer)
                
                # Clear existing dummy data for this institution
                cursor = analyzer.conn.cursor()
                cursor.execute('DELETE FROM institution_documents WHERE institution_id = ?', (institution_id,))
                
                # Generate enhanced documents with realistic patterns
                documents = generate_enhanced_institution_documents(institution_id, performance)
                
                # Insert into database with upload dates
                for doc in documents:
                    cursor.execute('''
                        INSERT INTO institution_documents 
                        (institution_id, document_name, document_type, status, upload_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (institution_id, doc['name'], doc['type'], doc['status'], doc['upload_date']))
                
                analyzer.conn.commit()
            
            st.session_state.enhanced_docs_generated = True
            print(f"✅ Generated enhanced document data for {len(institutions)} institutions")
            
        except Exception as e:
            print(f"Could not generate enhanced dummy documents: {e}")

def generate_enhanced_institution_documents(institution_id, performance):
    """Generate realistic document set with upload patterns and dates"""
    
    performance_score = performance['performance_score']
    institution_type = performance['institution_type']
    
    # Enhanced document templates with categories
    all_documents = {
        "mandatory_critical": [
            {"name": "Curriculum Framework 2023.pdf", "type": "curriculum_framework"},
            {"name": "Faculty Qualification Records.pdf", "type": "faculty_qualifications"},
            {"name": "Institutional Statutes.pdf", "type": "institutional_statutes"},
            {"name": "Annual Financial Statements.pdf", "type": "financial_statements"},
            {"name": "Academic Calendar 2023-24.pdf", "type": "academic_calendar"},
        ],
        "mandatory_important": [
            {"name": "Course Outlines and Syllabi.docx", "type": "course_outlines"},
            {"name": "Faculty Recruitment Policy.pdf", "type": "recruitment_policy"},
            {"name": "Organizational Structure.pdf", "type": "organizational_structure"},
            {"name": "Budget Allocation Document.pdf", "type": "budget_allocation"},
            {"name": "Learning Outcome Assessment.pdf", "type": "learning_outcomes"},
        ]
    }
    
    uploaded_documents = []
    
    # Define upload patterns based on performance
    if performance_score >= 8.5:  # Top performers
        pattern = {
            "mandatory_critical": 1.0,  # 100% uploaded
            "mandatory_important": 0.95, # 95% uploaded
        }
        
    elif performance_score >= 7.0:  # Good performers
        pattern = {
            "mandatory_critical": 1.0,   # 100% uploaded
            "mandatory_important": 0.85, # 85% uploaded
        }
        
    elif performance_score >= 5.5:  # Average performers
        pattern = {
            "mandatory_critical": 0.90,  # 90% uploaded
            "mandatory_important": 0.70, # 70% uploaded
        }
        
    else:  # Low performers
        pattern = {
            "mandatory_critical": 0.60,  # 60% uploaded
            "mandatory_important": 0.40, # 40% uploaded
        }
    
    # Generate upload dates (within last 6 months)
    base_date = datetime.now()
    
    for category, docs in all_documents.items():
        upload_probability = pattern[category]
        
        for doc in docs:
            if np.random.random() < upload_probability:
                # Document is uploaded - generate realistic upload date
                days_ago = np.random.randint(1, 180)  # Within last 6 months
                upload_date = base_date - timedelta(days=days_ago)
                
                uploaded_documents.append({
                    **doc,
                    "status": "Uploaded",
                    "upload_date": upload_date
                })
            else:
                # Document is pending
                uploaded_documents.append({
                    **doc,
                    "status": "Pending",
                    "upload_date": None
                })
    
    return uploaded_documents

def get_institution_performance(institution_id, analyzer):
    """Get institution performance data"""
    try:
        # Get the current year data (2023)
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        
        inst_data = current_year_data[current_year_data['institution_id'] == institution_id]
        
        if not inst_data.empty:
            return {
                'performance_score': float(inst_data['performance_score'].iloc[0]),
                'naac_grade': str(inst_data['naac_grade'].iloc[0]),
                'risk_level': str(inst_data['risk_level'].iloc[0]),
                'institution_name': str(inst_data['institution_name'].iloc[0]),
                'institution_type': str(inst_data['institution_type'].iloc[0])
            }
    except Exception as e:
        print(f"Error getting institution performance: {e}")
    
    # Return default values if data not found
    return {
        'performance_score': 5.0,
        'naac_grade': 'B',
        'risk_level': 'Medium Risk',
        'institution_name': 'Unknown Institution',
        'institution_type': 'General'
    }

def display_performance_context(performance, institution_id):
    """Display institution performance context"""
    
    st.subheader(f"🏛️ {performance['institution_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Color code based on performance
        if performance['performance_score'] >= 8.0:
            color = "green"
        elif performance['performance_score'] >= 6.0:
            color = "orange"
        else:
            color = "red"
        
        st.metric("Performance Score", f"{performance['performance_score']:.1f}/10")
    
    with col2:
        st.metric("NAAC Grade", performance['naac_grade'])
    
    with col3:
        risk_color = {
            "Low Risk": "green",
            "Medium Risk": "orange", 
            "High Risk": "red",
            "Critical Risk": "darkred"
        }.get(performance['risk_level'], "gray")
        
        st.metric("Risk Level", performance['risk_level'])
    
    with col4:
        st.metric("Institution Type", performance['institution_type'])
    
    # Performance interpretation
    if performance['performance_score'] >= 8.0:
        st.success("🎯 **High Performing Institution**: Expected to have comprehensive document submission")
    elif performance['performance_score'] >= 6.0:
        st.info("📊 **Medium Performing Institution**: Expected to have good document coverage")
    else:
        st.warning("⚠️ **Low Performing Institution**: May have incomplete document submission")

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
    
    # Store enhanced counts for analysis
    st.session_state.enhanced_document_counts = {
        'total_mandatory': total_mandatory,
        'uploaded_mandatory': uploaded_mandatory,
        'pending_mandatory': pending_mandatory,
        'uploaded_docs_data': uploaded_docs_data,
        'performance': performance
    }

def perform_enhanced_document_analysis(institution_id, approval_type, analyzer, performance):
    """Perform enhanced document analysis with performance impact"""
    
    counts = st.session_state.get('enhanced_document_counts', {})
    
    if not counts:
        st.error("No document data available for analysis")
        return
    
    total_mandatory = counts['total_mandatory']
    uploaded_mandatory = counts['uploaded_mandatory']
    pending_mandatory = counts['pending_mandatory']
    performance_data = counts['performance']
    
    # Calculate enhanced metrics
    mandatory_sufficiency = (uploaded_mandatory / total_mandatory) * 100 if total_mandatory > 0 else 0
    
    st.subheader("🎯 Enhanced Document Performance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mandatory Compliance", f"{mandatory_sufficiency:.1f}%")
    
    with col2:
        st.metric("Pending Documents", pending_mandatory)
    
    with col3:
        st.metric("Performance Score", f"{performance_data['performance_score']:.1f}/10")
    
    with col4:
        st.metric("Risk Level", performance_data['risk_level'])
    
    # Performance Impact Analysis
    st.subheader("📈 Document Upload Impact on Performance")
    
    # Show how document upload affects performance
    if mandatory_sufficiency >= 90:
        st.success("**🏆 EXCELLENT COMPLIANCE**: High document upload rate significantly boosts institutional performance")
        st.write("**Impact**: +0.3 points added to performance score")
    elif mandatory_sufficiency >= 70:
        st.info("**📊 GOOD COMPLIANCE**: Solid document submission supports institutional performance")
        st.write("**Impact**: +0.15 points added to performance score")
    elif mandatory_sufficiency >= 50:
        st.warning("**⚠️ AVERAGE COMPLIANCE**: Incomplete submission limits performance potential")
        st.write("**Impact**: No performance boost - missed opportunity")
    else:
        st.error("**🚨 POOR COMPLIANCE**: Significant document gaps negatively impact performance")
        st.write("**Impact**: -0.2 points deducted from performance score")

def get_document_requirements_by_parameters(approval_type):
    """Get document requirements organized by parameters"""
    
    base_requirements = {
        "new_approval": {
            "mandatory": {
                "Curriculum": [
                    "Curriculum framework documents",
                    "Course outlines and objectives",
                    "Academic calendar"
                ],
                "Faculty Resources": [
                    "Faculty qualification records",
                    "Recruitment policy documents"
                ],
                "Governance": [
                    "Institutional statutes",
                    "Organizational structure"
                ]
            },
            "supporting": {
                "Research": [
                    "Research publications",
                    "Project documentation"
                ],
                "Infrastructure": [
                    "Campus facility details",
                    "Laboratory equipment list"
                ]
            }
        },
        "renewal_approval": {
            "mandatory": {
                "Curriculum": [
                    "Updated curriculum framework",
                    "Academic performance reports"
                ],
                "Faculty": [
                    "Updated faculty records",
                    "Development reports"
                ]
            },
            "supporting": {
                "Research": [
                    "Recent publications",
                    "Research projects"
                ]
            }
        },
        "expansion_approval": {
            "mandatory": {
                "Infrastructure": [
                    "Expansion master plan",
                    "Additional facilities plan"
                ],
                "Faculty": [
                    "New faculty requirements",
                    "Recruitment plan"
                ]
            },
            "supporting": {
                "Financial": [
                    "Expansion budget",
                    "Funding plan"
                ]
            }
        }
    }
    
    return base_requirements.get(approval_type, base_requirements["new_approval"])
