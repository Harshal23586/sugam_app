# main.py
import streamlit as st
import sys
import os

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

def main():
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("🔄 Initializing AI Analytics System..."):
            st.session_state.analyzer = InstitutionalAIAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Create header
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image("assets/logo.jpg", width=200)
        except FileNotFoundError:
            st.warning("Logo file not found")
    with col2:
        st.title("सुगम")
        st.title("SUGAM - Smart University Governance and Approval Management")
    
    st.markdown("---")
    
    # Navigation based on user role
    if st.session_state.get('user_role') == 'Institution':
        # Institution user view
        user = st.session_state.get('institution_user')
        create_institution_dashboard(analyzer, user)
    else:
        # Admin/UGC view
        # Create sidebar navigation
        st.sidebar.title("🔧 Navigation")
        
        # User session info
        if st.session_state.get('institution_user'):
            user = st.session_state.institution_user
            st.sidebar.success(f"Logged in as: {user['contact_person']}")
            st.sidebar.info(f"Institution: {user['institution_name']}")
            
            if st.sidebar.button("🚪 Logout"):
                for key in ['institution_user', 'user_role']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            st.sidebar.info("UGC/AICTE Dashboard")
            if st.sidebar.button("🏛️ Institution Login"):
                st.session_state.show_institution_login = True
        
        # Main navigation options
        nav_options = [
            "📊 Performance Dashboard",
            "📋 Document Analysis", 
            "🧠 Intelligence Hub",
            "💾 Data Management",
            "📄 PDF Reports",
            "🌐 API Integration",
            "⚙️ System Settings",
            "🏛️ Institution Portal"
        ]
        
        selected_nav = st.sidebar.selectbox("Go to", nav_options)
        
        # Show selected module
        if selected_nav == "📊 Performance Dashboard":
            create_performance_dashboard(analyzer)
        
        elif selected_nav == "📋 Document Analysis":
            create_document_analysis_module(analyzer)
        
        elif selected_nav == "🧠 Intelligence Hub":
            create_institutional_intelligence_hub(analyzer)
        
        elif selected_nav == "💾 Data Management":
            create_data_management_module(analyzer)
        
        elif selected_nav == "📄 PDF Reports":
            create_pdf_report_module(analyzer)
        
        elif selected_nav == "🌐 API Integration":
            create_api_documentation()
        
        elif selected_nav == "⚙️ System Settings":
            create_system_settings_module(analyzer)
        
        elif selected_nav == "🏛️ Institution Portal":
            create_institution_login(analyzer)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("© 2025 UGC/AICTE - SUGAM System")
    with col2:
        st.caption("Version 2.0.0 | Smart India Hackathon 2025")
    with col3:
        st.caption("For support: support@sugam.ugc.gov.in")

if __name__ == "__main__":
    main()

