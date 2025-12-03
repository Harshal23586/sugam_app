import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from core.analyzer import InstitutionalAIAnalyzer
from modules.dashboard import create_performance_dashboard
from modules.document_analysis import create_document_analysis_module
from modules.intelligence_hub import create_institutional_intelligence_hub
from modules.data_management import create_data_management_module
from modules.api_documentation import create_api_documentation
from modules.pdf_reports import create_pdf_report_module
from institution.auth import create_institution_login
from institution.dashboard import create_institution_dashboard
from utils.config import initialize_session_state
initialize_session_state()


# Page configuration
st.set_page_config(
    page_title="SUGAM - Smart University Governance and Approval Management",
    page_icon="assets/logo.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.institution_user = None
    st.session_state.user_role = None
    st.session_state.rag_analysis = None
    st.session_state.selected_institution = None

def main():
    # Create header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image("assets/logo.jpg", width=200)
        except FileNotFoundError:
            st.warning("Logo file not found. Please ensure assets/logo.jpg exists.")
            
    with col2:
        st.title("सुगम")
        st.title("SUGAM - Smart University Governance and Approval Management")

    st.markdown("---")
    
    # Initialize analyzer
    analyzer = InstitutionalAIAnalyzer()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
    
    # Check if user is logged in
        if st.session_state.user_role == "Institution":
            # Institution user navigation
            tabs = ["🏠 Home", "🏛️ Institution Portal", "📊 Analytics", "📄 Reports", "⚙️ Settings"]
        else:
            # Public/non-logged in user navigation
            tabs = ["🏠 Home", "📊 Analytics", "📄 Reports", "🔐 Login"]
        
        # Create navigation
        selected_tab = st.radio("Go to", tabs, index=tabs.index(st.session_state.active_tab) if st.session_state.active_tab in tabs else 0)
    
        # Update active tab
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab
            st.rerun()
    
    if st.session_state.institution_user:
        # Institution user navigation
        user = st.session_state.institution_user
        st.sidebar.success(f"Logged in as: {user.get('contact_person', 'User')}")
        
        # Institution-specific tabs
        institution_tabs = [
            "🏛️ Institution Dashboard",
            "📤 Document Upload",
            "📝 Basic Data Submission",
            "🏛️ Systematic Data Form",
            "📊 My Submissions",
            "🔄 Approval Workflow",
            "📋 Requirements Guide"
        ]
        
        selected_tab = st.sidebar.radio("Go to", institution_tabs)
        
        if selected_tab == "🏛️ Institution Dashboard":
            create_institution_dashboard(analyzer, user)
        elif selected_tab == "📤 Document Upload":
            from institution.documents import create_institution_document_upload
            create_institution_document_upload(analyzer, user)
        elif selected_tab == "📝 Basic Data Submission":
            from institution.forms import create_institution_data_submission
            create_institution_data_submission(analyzer, user)
        elif selected_tab == "🏛️ Systematic Data Form":
            from institution.forms import create_systematic_data_submission_form
            create_systematic_data_submission_form(analyzer, user)
        elif selected_tab == "📊 My Submissions":
            from institution.submissions import create_institution_submissions_view
            create_institution_submissions_view(analyzer, user)
        elif selected_tab == "🔄 Approval Workflow":
            from institution.dashboard import create_institution_approval_workflow
            create_institution_approval_workflow(analyzer, user)
        elif selected_tab == "📋 Requirements Guide":
            from institution.dashboard import create_institution_requirements_guide
            create_institution_requirements_guide(analyzer)
            
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.institution_user = None
            st.session_state.user_role = None
            st.rerun()
            
    else:
        # Public/UGC navigation
        navigation_options = [
            "🏠 Performance Dashboard",
            "📋 Document Analysis",
            "🧠 Intelligence Hub",
            "💾 Data Management",
            "🌐 API Integration",
            "📄 PDF Reports",
            "🏛️ Institution Portal"
        ]
        
        selected_option = st.sidebar.radio("Select Module", navigation_options)
        
        if selected_option == "🏠 Performance Dashboard":
            create_performance_dashboard(analyzer)
        elif selected_option == "📋 Document Analysis":
            create_document_analysis_module(analyzer)
        elif selected_option == "🧠 Intelligence Hub":
            create_institutional_intelligence_hub(analyzer)
        elif selected_option == "💾 Data Management":
            create_data_management_module(analyzer)
        elif selected_option == "🌐 API Integration":
            create_api_documentation()
        elif selected_option == "📄 PDF Reports":
            create_pdf_report_module(analyzer)
        elif selected_option == "🏛️ Institution Portal":
            create_institution_login(analyzer)

if __name__ == "__main__":
    main()

