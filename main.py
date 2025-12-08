# main.py
import streamlit as st
import sys
import os
from datetime import datetime
from rag_core import create_rag_validation_dashboard

# Add project root to path

# Page configuration
st.set_page_config(
    page_title="SUGAM - Smart Unified Governance and Approval Management",
    page_icon="assets/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from modules.rag_core import InstitutionalRAGSystem
from modules.rag_dashboard import create_rag_dashboard

def main():
    # Safe session state initialization
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'ugc_aicte_user' not in st.session_state:
        st.session_state.ugc_aicte_user = None
    
    # Check if institution user is logged in (using our simple auth)
    if st.session_state.institution_user is not None:
        try:
            analyzer = InstitutionalAIAnalyzer()
            # Create a proper user object/dictionary that create_institution_dashboard expects
            user_info = {
                'username': st.session_state.institution_user,
                'role': 'institution',
                'institution_id': 'INST001',  # Default institution ID
                'name': 'Sample Institution'
            }
            create_institution_dashboard(analyzer, user_info)
            
            # Add logout button in sidebar if it exists
            if st.sidebar.button("🚪 Logout"):
                st.session_state.institution_user = None
                st.session_state.user_role = None
                st.rerun()
            return
        except Exception as e:
            st.error(f"❌ System initialization error: {str(e)}")
            st.write("Debug: Attempting alternative dashboard call...")
            
            # Try alternative approach
            try:
                # Try calling without user_info
                analyzer = InstitutionalAIAnalyzer()
                create_institution_dashboard(analyzer)
                return
            except Exception as e2:
                st.error(f"❌ Alternative approach also failed: {str(e2)}")
                # If error occurs, clear session and show landing page
                st.session_state.institution_user = None
                st.session_state.user_role = None
    
    # Check if UGC/AICTE user is logged in
    if st.session_state.ugc_aicte_user is not None:
        try:
            analyzer = InstitutionalAIAnalyzer()
            show_main_application(analyzer)
            return
        except Exception as e:
            st.error(f"❌ System initialization error: {str(e)}")
            st.session_state.ugc_aicte_user = None
            st.session_state.user_role = None
    
    # LANDING PAGE - No dashboard data shown here
    show_landing_page()

def show_landing_page():
    """Display the clean landing page with authentication options"""
    
    # Main header with logo
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Display logo with 200px width
        try:
            st.image("assets/logo.jpg", width=200)
        except:
            # Try different common paths
            try:
                st.image("logo.png", width=200)
            except:
                try:
                    st.image("logo.jpeg", width=200)
                except:
                    # Fallback placeholder
                    st.markdown("""
                    <div style="width: 200px; height: 200px; background-color: #0047AB; 
                                color: white; display: flex; align-items: center; 
                                justify-content: center; border-radius: 10px; font-size: 24px;">
                        <strong>SUGAM</strong>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h1 class="main-header">सुगम - SUGAM - Smart Unified Governance and Approval Management</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    # System overview from PDF report
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #f0f7ff; border-radius: 10px; margin-bottom: 20px;'>
        <h4>🏛️ National Education Policy 2020 (NEP 2020) Implementation Platform</h4>
        <p>This AI-powered platform supports the transformative reforms for strengthening assessment and 
        accreditation of Higher Education Institutions in India as per the Dr. Radhakrishnan Committee Report.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key features from the PDF
        st.subheader("📋 Key Features")
        
        features = [
            "✅ **Binary Accreditation System**: Transition from 8-point grading to 'Accredited', 'Awaiting Accreditation', 'Not Accredited'",
            "✅ **Level-Based Excellence**: Institutions graded from Level 1 to Level 5 for National & Global Excellence",
            "✅ **Unified Data Platform**: 'One Nation One Data' architecture for centralized data management",
            "✅ **Technology-Driven Assessment**: Minimize manual involvement through AI and automation",
            "✅ **Composite Assessment**: Amalgamate programme and institutional accreditation",
            "✅ **Stakeholder Crowdsourcing**: Enhanced verification through stakeholder participation",
            "✅ **Choice-Based Ranking**: Customizable ranking system for diverse user needs"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #fff3cd; border-radius: 10px; margin-bottom: 20px;'>
        <h4>🔐 Secure Access</h4>
        <p>Authorized access only for registered institutions and UGC/AICTE personnel.</p>
        <p>All activities are logged and monitored as per MoE guidelines.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats about Indian HEI landscape from PDF
        st.subheader("🇮🇳 Indian HEI Landscape")
        
        stats = {
            "Total HEIs in India": "50,000+",
            "NAAC Accredited Institutions": "36.67% of Universities",
            "NAAC Accredited Colleges": "21.64% of Colleges",
            "IITs in Global Rankings": "8 in Top 400 QS Ranking",
            "Washington Accord Member": "Yes (since 2014)"
        }
        
        for key, value in stats.items():
            st.metric(key, value)
    
    # Authentication Section
    st.markdown("---")
    st.subheader("🔐 System Access")
    
    # Create two columns for different login types
    login_col1, login_col2 = st.columns(2)
    
    with login_col1:
        st.markdown("### 🏫 Institution Login")
        st.info("For Higher Education Institutions (HEIs)")
        
        with st.form("institution_login"):
            inst_username = st.text_input("Username", placeholder="Enter institution username")
            inst_password = st.text_input("Password", type="password", placeholder="Enter password")
            inst_submit = st.form_submit_button("Login as Institution")
            
            if inst_submit:
                # Default credentials for demonstration
                if inst_username == "institute" and inst_password == "institute123":
                    st.session_state.institution_user = inst_username
                    st.session_state.user_role = "Institution"
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use: institute / institute123")
    
    with login_col2:
        st.markdown("### 🏛️ UGC/AICTE Login")
        st.info("For Regulatory Authorities")
        
        with st.form("ugc_aicte_login"):
            ugc_username = st.text_input("Username", placeholder="Enter UGC/AICTE username")
            ugc_password = st.text_input("Password", type="password", placeholder="Enter password")
            ugc_submit = st.form_submit_button("Login as UGC/AICTE")
            
            if ugc_submit:
                # Default credentials for demonstration
                if ugc_username == "ugc" and ugc_password == "ugc123":
                    st.session_state.ugc_aicte_user = ugc_username
                    st.session_state.user_role = "UGC/AICTE Officer"
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use: ugc / ugc123")
    
    # Display PDF report highlights
    st.markdown("---")
    st.subheader("📘 Transformative Reforms Overview")
    
    tab1, tab2, tab3 = st.tabs(["Committee Recommendations", "Global Best Practices", "Implementation Timeline"])
    
    with tab1:
        recommendations = [
            "1. **Adapted Binary Accreditation**: Three categories instead of two for better granularity",
            "2. **Maturity-Based Levels**: Institutions progress from Level 1 to Level 5 excellence",
            "3. **Composite Assessment**: Combine institutional and programme accreditation",
            "4. **Mentorship Program**: Accredited institutions mentor non-accredited ones",
            "5. **Simplified Process**: Reduced periodicity and simplified first-cycle accreditation",
            "6. **Inclusive Approach**: All HEIs including IITs under unified system",
            "7. **Category-Based Assessment**: Consider heterogeneity of HEIs"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    
    with tab2:
        practices = [
            "• **Mandatory Accreditation**: Required for degree awarding in many countries",
            "• **Binary Outcomes**: Acceptance or rejection with different shades",
            "• **Student Learning Focus**: Measured through exam scripts and interviews",
            "• **Public Disclosure**: All important documents in public domain",
            "• **Stakeholder Surveys**: Anonymous feedback collection",
            "• **High Engagement**: Mature mentoring at all stages"
        ]
        
        for practice in practices:
            st.markdown(practice)
    
    with tab3:
        timeline_data = {
            "Stage I (Short-Term)": [
                "Implement 'One Nation One Data' platform",
                "Launch Binary and Maturity-Based Graded Accreditation",
                "Capture common data used by all agencies"
            ],
            "Stage II (Medium-Term)": [
                "Expand to entire super set of data",
                "Implement stakeholder crowdsourcing",
                "Full technology integration"
            ]
        }
        
        for stage, tasks in timeline_data.items():
            st.markdown(f"**{stage}**")
            for task in tasks:
                st.markdown(f"• {task}")
    
    # Alternative: Use the original institution login module
    st.markdown("---")
    st.subheader("🏫 Alternative Institution Access")
    
    with st.expander("Use Original Institution Login System"):
        try:
            analyzer = InstitutionalAIAnalyzer()
            create_institution_login(analyzer)
        except Exception as e:
            st.warning(f"Original login system unavailable: {str(e)}")
    
    # Footer with current date
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>Based on Dr. Radhakrishnan Committee Report (November 2023)</strong> | Ministry of Education, Government of India</p>
    <p>SUGAM Platform v2.0 | Access restricted to authorized personnel | {datetime.now().strftime("%d %B %Y")}</p>
    </div>
    """, unsafe_allow_html=True)

def show_main_application(analyzer):
    """Show the main application after UGC/AICTE login"""
    
    # Display system stats in sidebar
    try:
        total_institutions = analyzer.historical_data['institution_id'].nunique()
        total_years = analyzer.historical_data['year'].nunique()
        total_records = len(analyzer.historical_data)
        
        st.sidebar.success(f"📊 Data: {total_institutions} institutes × {total_years} years")
        st.sidebar.info(f"📈 Total Records: {total_records}")
        
        if total_institutions == 20 and total_years == 10 and total_records == 200:
            st.sidebar.success("✅ 20×10 specification verified")
        else:
            st.sidebar.warning(f"⚠️ Data mismatch: Expected 20×10=200, Got {total_institutions}×{total_years}={total_records}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Data verification issue: {str(e)}")
    
    # SINGLE sidebar navigation section for UGC/AICTE users
    st.sidebar.title("🧭 Navigation Panel")
    st.sidebar.markdown("---")
    
    # User info and logout
    st.sidebar.markdown(f"**👤 Logged in as:** {st.session_state.user_role}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.ugc_aicte_user = None
        st.session_state.user_role = None
        st.rerun()
    
    st.sidebar.markdown("### AI Modules")
    
    app_mode = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "📊 Performance Dashboard",
            "📋 Document Analysis", 
            "🤖 Intelligence Hub",
            "🔍 RAG Data Management",
            "🔍 Document-Form Validation",
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

    elif app_mode == "🌐 API Integration":
        create_api_documentation()

    elif app_mode == "📄 PDF Reports":
        create_pdf_report_module(analyzer)
    
    elif app_mode == "🔍 Document-Form Validation":
        create_rag_validation_dashboard(analyzer)
        
        # Display system information
        st.subheader("System Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Records", len(analyzer.historical_data))
        with col2:
            st.metric("Unique Institutions", analyzer.historical_data['institution_id'].nunique())
        with col3:
            st.metric("Data Years", f"{analyzer.historical_data['year'].min()}-{analyzer.historical_data['year'].max()}")
    
    # Footer for main application
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>UGC/AICTE Institutional Analytics Platform</strong> | AI-Powered Decision Support System</p>
    <p>Version 2.0 | For authorized use only | Data last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


