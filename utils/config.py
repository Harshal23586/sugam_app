# utils/config.py
"""
Configuration and session state management
"""

import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    # User authentication
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Navigation
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "🏠 Home"
    if 'selected_institution' not in st.session_state:
        st.session_state.selected_institution = None
    
    # Document analysis
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    if 'rag_analysis' not in st.session_state:
        st.session_state.rag_analysis = None
    
    # Data management
    if 'enhanced_docs_generated' not in st.session_state:
        st.session_state.enhanced_docs_generated = False
    if 'selected_for_comparison' not in st.session_state:
        st.session_state.selected_for_comparison = []
    
    # Form state
    if 'form_submission_count' not in st.session_state:
        st.session_state.form_submission_count = 0
    
    return True

def clear_user_session():
    """Clear user session data"""
    st.session_state.institution_user = None
    st.session_state.user_role = None
    st.success("Logged out successfully!")

def get_user_info():
    """Get current user information"""
    return {
        'user': st.session_state.get('institution_user'),
        'role': st.session_state.get('user_role'),
        'authenticated': st.session_state.get('institution_user') is not None
    }

def set_active_tab(tab_name):
    """Set active tab for navigation"""
    st.session_state.active_tab = tab_name
