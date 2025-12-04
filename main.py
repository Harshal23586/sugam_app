# main.py
import streamlit as st
import sys
import os
from datetime import datetime
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="SUGAM - Smart University Governance and Approval Management",
    page_icon="assets/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from core.analyzer import InstitutionalAIAnalyzer
from modules.dashboard import create_performance_dashboard
from modules.document_analysis import create_document_analysis_module
from modules.intelligence_hub import create_institutional_intelligence_hub
from modules.data_management import create_data_management_module
from modules.api_documentation import create_api_documentation
from modules.pdf_reports import create_pdf_report_module
from modules.system_settings import create_system_settings
from institution.auth import create_institution_login
from institution.dashboard import create_institution_dashboard
from modules.rag_core import InstitutionalDocument
from modules.rag_dashboard import create_rag_dashboard

def main():
    # Safe session state initialization
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Initialize analytics engine with error handling
    try:
        analyzer = InstitutionalAIAnalyzer()
        
        # Verify data specifications
        total_institutions = analyzer.historical_data['institution_id'].nunique()
        total_years = analyzer.historical_data['year'].nunique()
        total_records = len(analyzer.historical_data)
        
        st.sidebar.success(f"📊 Data: {total_institutions} institutes × {total_years} years")
        st.sidebar.info(f"📈 Total Records: {total_records}")
        
        # Show data verification
        if total_institutions == 20 and total_years == 10 and total_records == 200:
            st.sidebar.success("✅ 20×10 specification verified")
        else:
            st.sidebar.warning(f"⚠️ Data mismatch: Expected 20×10=200, Got {total_institutions}×{total_years}={total_records}")
            
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.stop()
    
    # Check if institution user is logged in
    if st.session_state.institution_user is not None:
        create_institution_dashboard(analyzer, st.session_state.institution_user)
        if st.sidebar.button("🚪 Logout"):
            st.session_state.institution_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    # Main header and system overview
    #st.markdown('<h1 class="main-header">सुगम - SUGAM - Smart University Governance and Approval Management</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    # System overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🚀 System Overview</h4>
        <p>This AI-powered platform automates the analysis of institutional historical data, performance metrics, 
        and document compliance for UGC and AICTE approval processes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>🔒 Secure Access</h4>
        <p>Authorized UGC/AICTE personnel and registered institutions only. All activities are logged and monitored.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.success("✅ AI Analytics System Successfully Initialized!")
    
    # Display quick stats
    st.subheader("📈 System Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_institutions = len(analyzer.historical_data['institution_id'].unique())
        st.metric("Total Institutions", total_institutions)
    
    with col2:
        years_data = len(analyzer.historical_data['year'].unique())
        st.metric("Years of Data", years_data)
    
    with col3:
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        if len(current_year_data) > 0:
            avg_performance = current_year_data['performance_score'].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}/10")
        else:
            st.metric("Avg Performance Score", "N/A")
    
    with col4:
        if len(current_year_data) > 0:
            approval_ready = (current_year_data['performance_score'] >= 6.0).sum()
            st.metric("Approval Ready", approval_ready)
        else:
            st.metric("Approval Ready", "N/A")
    
    # SINGLE sidebar navigation section - REMOVE APPROVAL WORKFLOW FOR NON-INSTITUTION ROLES
    st.sidebar.title("🧭 Navigation Panel")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Authentication")
    
    user_role = st.sidebar.selectbox(
        "Select Your Role",
        ["Institution", "UGC Officer", "AICTE Officer", "System Admin", "Review Committee"]
    )
    
    if user_role == "Institution":
        create_institution_login(analyzer)
        return
    
    # For other roles, show the AI modules (WITHOUT APPROVAL WORKFLOW)
    st.sidebar.markdown("### AI Modules")
    
    app_mode = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "📊 Performance Dashboard",
            "📋 Document Analysis", 
            "🤖 Intelligence Hub",
            "🔍 RAG Data Management",
            "💾 Data Management",
            "📄 PDF Reports",
            "🌐 API Integration",
            "⚙️ System Settings"
        ]
    )
    
    # Route to selected module
    if app_mode == "📊 Performance Dashboard":
        create_performance_dashboard(analyzer)
    
    elif app_mode == "📋 Document Analysis":
        create_document_analysis_module(analyzer)
    
    elif app_mode == "🤖 Intelligence Hub":
        create_institutional_intelligence_hub(analyzer)
    
    elif app_mode == "🔍 RAG Data Management":
        create_rag_dashboard(analyzer)
    
    elif app_mode == "💾 Data Management":
        create_data_management_module(analyzer)
    
    elif app_mode == "⚙️ System Settings":
        create_system_settings(analyzer)

    # Add to your main navigation in main() function:
    elif app_mode == "🌐 API Integration":
        create_api_documentation()

    # Add to routing logic:
    elif app_mode == "📄 PDF Reports":
        create_pdf_report_module(analyzer)
        
        st.subheader("System Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Records", len(analyzer.historical_data))
        with col2:
            st.metric("Unique Institutions", analyzer.historical_data['institution_id'].nunique())
        with col3:
            st.metric("Data Years", f"{analyzer.historical_data['year'].min()}-{analyzer.historical_data['year'].max()}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>UGC/AICTE Institutional Analytics Platform</strong> | AI-Powered Decision Support System</p>
    <p>Version 2.0 | For authorized use only | Data last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
if __name__ == "__main__":
    main()






