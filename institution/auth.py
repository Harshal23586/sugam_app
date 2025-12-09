# institution/auth.py
"""
Institution Authentication Module

Handles institution user login and registration with enhanced security
and user experience features.
"""

import streamlit as st
import hashlib
from typing import Dict, Optional

def create_institution_login(analyzer):
    """
    Create institution login and registration interface
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
    """
    st.header("🔐 Institution Portal Login")
    
    # Create two columns for login and registration
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Login Section
        st.subheader("📋 Existing Institution Users")
        
        with st.form("institution_login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Your institution username"
            )
            
            password = st.text_input(
                "Password", 
                type="password",
                placeholder="Enter your password",
                help="Your institution password"
            )
            
            # Remember me option
            remember_me = st.checkbox("Remember me", value=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_submitted = st.form_submit_button(
                    "🚀 Login",
                    type="primary",
                    use_container_width=True
                )
            with col_btn2:
                if st.form_submit_button(
                    "🔓 Forgot Password?",
                    type="secondary",
                    use_container_width=True
                ):
                    st.info("Password reset feature coming soon!")
            
            if login_submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        user = analyzer.authenticate_institution_user(username, password)
                        if user:
                            # Store user in session state
                            st.session_state.institution_user = user
                            st.session_state.user_role = "Institution"
                            
                            # Show success message
                            st.success(f"✅ Welcome back, {user['contact_person']}!")
                            st.success(f"Institution: {user['institution_name']}")
                            
                            # Add a small delay for better UX
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password. Please try again.")
    
    with col2:
        # Registration Section
        st.subheader("📝 New Institution Registration")
        
        with st.form("institution_registration_form"):
            # Get available institutions for registration
            available_institutions = get_available_institutions(analyzer)
            
            if available_institutions.empty:
                st.warning("No institutions available for registration")
                return
            
            # Institution selection
            selected_institution = st.selectbox(
                "Select Your Institution",
                available_institutions['institution_id'].tolist(),
                format_func=lambda x: available_institutions[
                    available_institutions['institution_id'] == x
                ]['institution_name'].iloc[0],
                help="Select your institution from the list"
            )
            
            # User details
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                new_username = st.text_input(
                    "Choose Username",
                    placeholder="e.g., admin_2024",
                    help="Choose a unique username"
                )
            
            with col_reg2:
                contact_person = st.text_input(
                    "Contact Person Name",
                    placeholder="Full name of contact person",
                    help="Primary contact person for the institution"
                )
            
            col_reg3, col_reg4 = st.columns(2)
            with col_reg3:
                new_password = st.text_input(
                    "Choose Password",
                    type="password",
                    placeholder="Strong password",
                    help="Minimum 8 characters with mix of letters, numbers, symbols"
                )
            
            with col_reg4:
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Confirm password"
                )
            
            col_reg5, col_reg6 = st.columns(2)
            with col_reg5:
                email = st.text_input(
                    "Email Address",
                    placeholder="contact@institution.edu.in",
                    help="Official institution email"
                )
            
            with col_reg6:
                phone = st.text_input(
                    "Phone Number",
                    placeholder="+91-XXXXXXXXXX",
                    help="Contact number with country code"
                )
            
            # Terms and conditions
            accept_terms = st.checkbox(
                "I agree to the terms and conditions",
                value=False
            )
            
            register_submitted = st.form_submit_button(
                "📋 Register Institution Account",
                type="primary",
                use_container_width=True
            )
            
            if register_submitted:
                # Validate inputs
                validation_result = validate_registration_inputs(
                    new_username, new_password, confirm_password,
                    contact_person, email, accept_terms
                )
                
                if validation_result["valid"]:
                    # Attempt to create user
                    success = analyzer.create_institution_user(
                        selected_institution,
                        new_username,
                        new_password,
                        contact_person,
                        email,
                        phone
                    )
                    
                    if success:
                        st.success("🎉 Institution account created successfully!")
                        st.info("You can now login with your credentials.")
                        
                        # Show next steps
                        with st.expander("📋 Next Steps"):
                            st.write("""
                            1. **Complete Profile**: Add additional institution details
                            2. **Upload Documents**: Start with mandatory documents
                            3. **Submit Data**: Fill institutional performance data
                            4. **Track Progress**: Monitor your approval status
                            """)
                    else:
                        st.error("❌ Username already exists. Please choose a different username.")
                else:
                    for error in validation_result["errors"]:
                        st.error(error)

def validate_contact(phone):
    if not phone:
        return False, "Phone number is required."

    # Remove spaces
    phone = phone.strip()

    # Allow +91 at start
    if phone.startswith("+91"):
        phone = phone[3:]

    # Allow leading zero
    if phone.startswith("0"):
        phone = contact[1:]

    if not phone.isdigit():
        return False, "Contact number must contain digits only."

    if len(phone) != 10:
        return False, "Contact number must be exactly 10 digits."

    return True, None


if st.button("Submit"):
    ok, msg = validate_contact(phone)
    if not ok:
        st.error(msg)
    else:
        st.success("Registration successful!")

def get_available_institutions(analyzer):
    """
    Get list of institutions available for registration
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
    
    Returns:
        DataFrame of available institutions
    """
    try:
        # Get current year institutions
        available_institutions = analyzer.historical_data[
            analyzer.historical_data['year'] == 2023
        ][['institution_id', 'institution_name']].drop_duplicates()
        
        # Filter out institutions that already have users
        cursor = analyzer.conn.cursor()
        cursor.execute('SELECT DISTINCT institution_id FROM institution_users')
        existing_institutions = [row[0] for row in cursor.fetchall()]
        
        # Filter to only show institutions without users
        if existing_institutions:
            available_institutions = available_institutions[
                ~available_institutions['institution_id'].isin(existing_institutions)
            ]
        
        return available_institutions
        
    except Exception as e:
        st.error(f"Error loading institutions: {str(e)}")
        return pd.DataFrame()

def validate_registration_inputs(username, password, confirm_password, 
                               contact_person, email, accept_terms):
    """
    Validate registration form inputs
    
    Args:
        username: Chosen username
        password: Chosen password
        confirm_password: Confirmed password
        contact_person: Contact person name
        email: Email address
        accept_terms: Whether terms are accepted
    
    Returns:
        Dictionary with validation result and errors
    """
    errors = []
    
    # Username validation
    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters long")
    
    # Password validation
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if password != confirm_password:
        errors.append("Passwords do not match")
    
    # Contact person validation
    if not contact_person or len(contact_person.strip()) < 2:
        errors.append("Please enter a valid contact person name")
    
    # Email validation (basic)
    if not email or "@" not in email or "." not in email:
        errors.append("Please enter a valid email address")
    
    # Terms acceptance
    if not accept_terms:
        errors.append("You must accept the terms and conditions")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def hash_password_simple(password: str) -> str:
    """
    Simple password hashing function
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def logout_user():
    """
    Logout current user and clear session state
    """
    if 'institution_user' in st.session_state:
        user_name = st.session_state.institution_user.get('contact_person', 'User')
        st.session_state.institution_user = None
        st.session_state.user_role = None
        st.success(f"👋 Goodbye, {user_name}! You have been logged out.")
        st.rerun()

# Add logout button to sidebar
def create_logout_button():
    """
    Create logout button for sidebar
    """
    if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
        logout_user()

if __name__ == "__main__":
    # Test the module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from core.analyzer import InstitutionalAIAnalyzer
    
    st.set_page_config(page_title="Institution Auth Test", layout="wide")
    analyzer = InstitutionalAIAnalyzer()
    create_institution_login(analyzer)

