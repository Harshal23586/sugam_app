# main.py
import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
#from rag_core import create_rag_validation_dashboard
#from modules.rag_core import create_rag_validation_dashboard
from modules.decision_tree_classifier import create_decision_tree_module
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
#from modules.rag_core import InstitutionalRAGSystem
from modules.rag_dashboard import create_rag_dashboard

# ============================================================================
# VALIDATION FUNCTIONS FOR ALL FORMS
# ============================================================================

def validate_username(username, user_type="institution"):
    """Validate username format"""
    errors = []
    
    if not username:
        errors.append("❌ Username is required")
        return errors
    
    if len(username) < 4:
        errors.append("❌ Username must be at least 4 characters long")
    
    if len(username) > 50:
        errors.append("❌ Username must be less than 50 characters")
    
    # Check for allowed characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', username):
        errors.append("❌ Username can only contain letters, numbers, underscores, hyphens, and dots")
    
    # Institution-specific validation
    if user_type == "institution":
        if not username.lower().startswith(('inst', 'college', 'univ', 'iit', 'nit')):
            errors.append("⚠️ Institution usernames should start with institution identifiers")
    
    # UGC/AICTE-specific validation
    if user_type == "ugc_aicte":
        if not username.lower().startswith(('ugc', 'aicte', 'admin', 'officer', 'inspector')):
            errors.append("⚠️ UGC/AICTE usernames should indicate regulatory authority role")
    
    return errors

def validate_password(password, user_type="institution"):
    """Validate password strength"""
    errors = []
    warnings = []
    
    if not password:
        errors.append("❌ Password is required")
        return errors, warnings
    
    # Length check
    if len(password) < 8:
        errors.append("❌ Password must be at least 8 characters long")
    
    if len(password) > 100:
        errors.append("❌ Password must be less than 100 characters")
    
    # Complexity checks
    if not re.search(r'[A-Z]', password):
        errors.append("❌ Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        errors.append("❌ Password must contain at least one lowercase letter")
    
    if not re.search(r'[0-9]', password):
        errors.append("❌ Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        warnings.append("⚠️ Consider adding special characters for stronger password")
    
    # Common password check
    common_passwords = ['password', '12345678', 'admin123', 'welcome', 'password123']
    if password.lower() in common_passwords:
        errors.append("❌ Password is too common. Please choose a stronger password")
    
    # Sequential characters check
    if re.search(r'(.)\1{3,}', password):
        warnings.append("⚠️ Avoid repeating the same character multiple times")
    
    # Institution-specific requirements
    if user_type == "institution":
        if len(password) < 10:
            warnings.append("⚠️ For institutions, consider using at least 10 characters")
    
    return errors, warnings

def validate_institution_selection(institution_id, available_institutions):
    """Validate institution selection"""
    errors = []
    
    if not institution_id:
        errors.append("❌ Institution selection is required")
        return errors
    
    if institution_id not in available_institutions:
        errors.append("❌ Selected institution is not available in the database")
    
    return errors

def validate_parameter_selection(selected_params, all_parameters):
    """Validate selected parameters for prediction"""
    errors = []
    warnings = []
    
    if not selected_params:
        errors.append("❌ At least one parameter must be selected for prediction")
        return errors, warnings
    
    # Check for reasonable parameter combinations
    if len(selected_params) < 3:
        warnings.append("⚠️ For better prediction accuracy, select at least 3 parameters")
    
    # Check for invalid parameters
    invalid_params = [p for p in selected_params if p not in all_parameters]
    if invalid_params:
        errors.append(f"❌ Invalid parameters selected: {', '.join(invalid_params[:3])}")
    
    # Check for parameter categories coverage
    param_categories = {
        'academic': ['student_faculty_ratio', 'phd_faculty_ratio', 'research_publications'],
        'financial': ['research_grants_amount', 'financial_stability_score'],
        'infrastructure': ['digital_infrastructure_score', 'library_volumes', 'laboratory_equipment_score'],
        'governance': ['compliance_score', 'administrative_efficiency'],
        'outcomes': ['placement_rate', 'higher_education_rate'],
        'community': ['community_projects', 'rural_outreach_score', 'inclusive_education_index']
    }
    
    covered_categories = set()
    for param in selected_params:
        for category, params in param_categories.items():
            if param in params:
                covered_categories.add(category)
    
    if len(covered_categories) < 2:
        warnings.append("⚠️ Consider selecting parameters from different categories for comprehensive analysis")
    
    return errors, warnings

def validate_numeric_input(value, param_name, min_val=None, max_val=None):
    """Validate numeric input values"""
    errors = []
    
    try:
        num_value = float(value)
        
        if min_val is not None and num_value < min_val:
            errors.append(f"❌ {param_name} must be at least {min_val}")
        
        if max_val is not None and num_value > max_val:
            errors.append(f"❌ {param_name} must be at most {max_val}")
        
        # Check for reasonable ranges based on parameter type
        if 'ratio' in param_name or 'rate' in param_name:
            if not (0 <= num_value <= 100):
                errors.append(f"❌ {param_name} should be between 0 and 100")
        
        if 'score' in param_name:
            if not (0 <= num_value <= 10):
                errors.append(f"❌ {param_name} should be between 0 and 10")
        
    except (ValueError, TypeError):
        errors.append(f"❌ {param_name} must be a valid number")
    
    return errors

def validate_year_input(year, min_year=2000, max_year=None):
    """Validate year input"""
    errors = []
    
    if not year:
        errors.append("❌ Year is required")
        return errors
    
    try:
        year_int = int(year)
        
        if year_int < min_year:
            errors.append(f"❌ Year must be {min_year} or later")
        
        if max_year and year_int > max_year:
            errors.append(f"❌ Year cannot be later than {max_year}")
        
        if year_int > datetime.now().year:
            errors.append("❌ Year cannot be in the future")
            
    except (ValueError, TypeError):
        errors.append("❌ Year must be a valid number")
    
    return errors

def validate_email(email):
    """Validate email format"""
    errors = []
    
    if not email:
        errors.append("❌ Email is required")
        return errors
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        errors.append("❌ Please enter a valid email address")
    
    # Check for common email issues
    if '..' in email:
        errors.append("❌ Email cannot contain consecutive dots")
    
    if email.startswith('.') or email.endswith('.'):
        errors.append("❌ Email cannot start or end with a dot")
    
    return errors

def validate_phone(phone):
    """Validate phone number"""
    errors = []
    
    if not phone:
        return errors  # Phone is optional
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    if len(digits_only) < 10:
        errors.append("❌ Phone number must have at least 10 digits")
    
    if len(digits_only) > 15:
        errors.append("❌ Phone number is too long")
    
    # Check for valid country code if present
    if digits_only.startswith('91') and len(digits_only) != 12:
        errors.append("❌ Indian phone numbers should be 10 digits (excluding country code)")
    
    return errors

def validate_file_upload(file, allowed_extensions=['csv', 'xlsx', 'xls', 'pdf', 'txt']):
    """Validate uploaded file"""
    errors = []
    
    if not file:
        errors.append("❌ Please select a file to upload")
        return errors
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size > max_size:
        errors.append("❌ File size exceeds 10MB limit")
    
    # Check file extension
    file_extension = file.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        errors.append(f"❌ File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
    
    # Check for suspicious file names
    suspicious_patterns = ['.exe', '.bat', '.sh', '.js', '.php', '.py']
    for pattern in suspicious_patterns:
        if file.name.lower().endswith(pattern):
            errors.append(f"❌ File type {pattern} is not allowed for security reasons")
    
    return errors

def display_validation_errors(errors, warnings=None, success=None):
    """Display validation messages in a consistent format"""
    
    if errors:
        for error in errors:
            st.error(error)
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    if success:
        st.success(success)
    
    return len(errors) == 0

def validate_login_form(username, password, user_type="institution"):
    """Validate login form inputs"""
    all_errors = []
    all_warnings = []
    
    # Validate username
    username_errors = validate_username(username, user_type)
    all_errors.extend(username_errors)
    
    # Validate password
    password_errors, password_warnings = validate_password(password, user_type)
    all_errors.extend(password_errors)
    all_warnings.extend(password_warnings)
    
    return all_errors, all_warnings

# ============================================================================
# UPDATED PREDICTION FUNCTION WITH VALIDATION
# ============================================================================

def predict_performance_tab(analyzer):
    """Tab for predicting 5-year performance using logistic regression"""
    st.header("📊 Institutional Performance Prediction (Logistic Regression)")
    
    # Load data
    data = analyzer.historical_data.copy()
    
    # Define available parameters
    all_parameters = [
        'student_faculty_ratio', 'phd_faculty_ratio', 'research_publications',
        'research_grants_amount', 'patents_filed', 'industry_collaborations',
        'digital_infrastructure_score', 'library_volumes', 'laboratory_equipment_score',
        'financial_stability_score', 'compliance_score', 'administrative_efficiency',
        'placement_rate', 'higher_education_rate', 'entrepreneurship_cell_score',
        'community_projects', 'rural_outreach_score', 'inclusive_education_index'
    ]
    
    # Create form for prediction configuration
    with st.form("prediction_config_form"):
        st.subheader("⚙️ Prediction Configuration")
        
        # Parameter selection with validation
        selected_params = st.multiselect(
            "Choose parameters for prediction model:",
            all_parameters,
            default=all_parameters[:8],
            help="Select at least 3 parameters from different categories for best results"
        )
        
        # Institution selection
        institutions = sorted(data['institution_id'].unique())
        selected_institution = st.selectbox(
            "Select institution to analyze:",
            institutions,
            index=0,
            help="Choose an institution from the available list"
        )
        
        # Additional options
        st.subheader("📈 Model Options")
        test_size = st.slider(
            "Test set size (%):",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
        
        random_seed = st.number_input(
            "Random seed:",
            min_value=0,
            max_value=1000,
            value=42,
            help="Random seed for reproducible results"
        )
        
        submit_button = st.form_submit_button("🚀 Run Prediction Analysis")
    
    if submit_button:
        # Perform validation
        validation_passed = True
        
        # Validate parameter selection
        param_errors, param_warnings = validate_parameter_selection(selected_params, all_parameters)
        if param_errors:
            display_validation_errors(param_errors)
            validation_passed = False
        if param_warnings:
            display_validation_errors([], param_warnings)
        
        # Validate institution selection
        inst_errors = validate_institution_selection(selected_institution, institutions)
        if inst_errors:
            display_validation_errors(inst_errors)
            validation_passed = False
        
        # Validate test size
        test_errors = validate_numeric_input(test_size, "Test size", 5, 50)
        if test_errors:
            display_validation_errors(test_errors)
            validation_passed = False
        
        if not validation_passed:
            st.error("❌ Please fix the validation errors above before proceeding.")
            return
        
        # If validation passed, proceed with prediction
        try:
            # Define target labels based on approval_recommendation
            def get_target_label(recommendation):
                recommendation_str = str(recommendation)
                if 'Provisional' in recommendation_str or 'Conditional' in recommendation_str:
                    return 1
                elif 'Approval' in recommendation_str:
                    return 1
                elif 'Rejection' in recommendation_str:
                    return 0
                else:
                    return 0
            
            # Create training data
            train_data = []
            for inst_id in data['institution_id'].unique():
                inst_data = data[data['institution_id'] == inst_id].sort_values('year')
                
                if len(inst_data) >= 6:
                    for i in range(len(inst_data) - 5):
                        window_data = inst_data.iloc[i:i+5]
                        features = {}
                        for param in selected_params:
                            if param in window_data.columns:
                                features[param] = window_data[param].mean()
                            else:
                                features[param] = 0
                        
                        future_data = inst_data.iloc[i+5:min(i+10, len(inst_data))]
                        if len(future_data) > 0:
                            future_labels = future_data['approval_recommendation'].apply(get_target_label)
                            target = 1 if future_labels.mean() > 0.5 else 0
                            features['target'] = target
                            train_data.append(features)
            
            if not train_data:
                st.error("❌ Insufficient data for prediction. Need more historical data.")
                return
            
            train_df = pd.DataFrame(train_data)
            
            # Train the model
            X = train_df[selected_params]
            y = train_df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=int(random_seed), stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=int(random_seed), max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make prediction for selected institution
            inst_data = data[data['institution_id'] == selected_institution].sort_values('year', ascending=False)
            
            if len(inst_data) >= 5:
                latest_5_years = inst_data.head(5).sort_values('year')
                prediction_features = {}
                for param in selected_params:
                    if param in latest_5_years.columns:
                        prediction_features[param] = latest_5_years[param].mean()
                    else:
                        prediction_features[param] = 0
                
                prediction_df = pd.DataFrame([prediction_features])
                prediction_scaled = scaler.transform(prediction_df)
                prediction = model.predict(prediction_scaled)[0]
                prediction_proba = model.predict_proba(prediction_scaled)[0]
                
                # Display results
                st.success("✅ Prediction completed successfully!")
                
                # Show results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Prediction Results")
                    if prediction == 1:
                        st.success("**Prediction: GOOD PERFORMANCE**")
                        st.metric("Confidence", f"{prediction_proba[1]*100:.1f}%")
                    else:
                        st.error("**Prediction: POOR PERFORMANCE**")
                        st.metric("Confidence", f"{prediction_proba[0]*100:.1f}%")
                
                with col2:
                    st.subheader("📈 Model Performance")
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Training Samples", len(X_train))
                    st.metric("Test Samples", len(X_test))
                
            else:
                st.error(f"❌ Selected institution needs at least 5 years of data. Currently has {len(inst_data)} years.")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# ============================================================================
# UPDATED MAIN FUNCTIONS WITH VALIDATION
# ============================================================================

def main():
    # Safe session state initialization
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'ugc_aicte_user' not in st.session_state:
        st.session_state.ugc_aicte_user = None
    
    # Check if institution user is logged in
    if st.session_state.institution_user is not None:
        try:
            analyzer = InstitutionalAIAnalyzer()
            user_info = {
                'username': st.session_state.institution_user,
                'role': 'institution',
                'institution_id': 'INST001',
                'name': 'Sample Institution'
            }
            create_institution_dashboard(analyzer, user_info)
            
            if st.sidebar.button("🚪 Logout"):
                st.session_state.institution_user = None
                st.session_state.user_role = None
                st.rerun()
            return
        except Exception as e:
            st.error(f"❌ System initialization error: {str(e)}")
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
    
    # Show landing page with login forms
    show_landing_page()

def show_landing_page():
    """Display the clean landing page with authentication options"""
    
    # Main header with logo
    col1, col2 = st.columns([1, 4])
    
    with col1:
        try:
            st.image("assets/logo.jpg", width=200)
        except:
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
    
    # System overview
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
        
        # Key features with validation note
        st.subheader("📋 Key Features")
        features = [
            "✅ **Binary Accreditation System**: Transition from 8-point grading to 'Accredited', 'Awaiting Accreditation', 'Not Accredited'",
            "✅ **Level-Based Excellence**: Institutions graded from Level 1 to Level 5 for National & Global Excellence",
            "✅ **Unified Data Platform**: 'One Nation One Data' architecture for centralized data management",
            "✅ **Technology-Driven Assessment**: Minimize manual involvement through AI and automation",
            "✅ **Composite Assessment**: Amalgamate programme and institutional accreditation",
            "✅ **Stakeholder Crowdsourcing**: Enhanced verification through stakeholder participation",
            "✅ **Choice-Based Ranking**: Customizable ranking system for diverse user needs",
            "✅ **AI Performance Prediction**: Logistic regression model to predict 5-year institutional performance",
            "✅ **Form Validation**: Comprehensive input validation for all forms ensuring data quality"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #fff3cd; border-radius: 10px; margin-bottom: 20px;'>
        <h4>🔐 Secure Access with Validation</h4>
        <p>Authorized access only for registered institutions and UGC/AICTE personnel.</p>
        <p>All inputs are validated for security and data quality.</p>
        <p>All activities are logged and monitored as per MoE guidelines.</p>
        </div>
        """, unsafe_allow_html=True)
        
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
    
    # Authentication Section with validated forms
    st.markdown("---")
    st.subheader("🔐 System Access with Input Validation")
    
    # Create two columns for different login types
    login_col1, login_col2 = st.columns(2)
    
    with login_col1:
        st.markdown("### 🏫 Institution Login")
        st.info("For Higher Education Institutions (HEIs)")
        
        with st.form("institution_login", clear_on_submit=False):
            inst_username = st.text_input("Username", 
                                        placeholder="Enter institution username (min. 4 chars)",
                                        key="inst_username")
            inst_password = st.text_input("Password", 
                                        type="password", 
                                        placeholder="Enter password (min. 8 chars with uppercase, lowercase, number)",
                                        key="inst_password")
            
            # Additional fields for institution registration
            with st.expander("Additional Information (Optional)"):
                inst_email = st.text_input("Email", 
                                         placeholder="Enter institution email",
                                         key="inst_email")
                inst_contact = st.text_input("Contact Person", 
                                           placeholder="Enter contact person name",
                                           key="inst_contact")
            
            inst_submit = st.form_submit_button("✅ Login as Institution")
            
            if inst_submit:
                # Perform validation
                validation_passed = True
                
                # Validate username
                username_errors = validate_username(inst_username, "institution")
                if username_errors:
                    display_validation_errors(username_errors)
                    validation_passed = False
                
                # Validate password
                password_errors, password_warnings = validate_password(inst_password, "institution")
                if password_errors:
                    display_validation_errors(password_errors)
                    validation_passed = False
                if password_warnings:
                    display_validation_errors([], password_warnings)
                
                # Validate email if provided
                if inst_email:
                    email_errors = validate_email(inst_email)
                    if email_errors:
                        display_validation_errors(email_errors)
                        validation_passed = False
                
                if validation_passed:
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
        
        with st.form("ugc_aicte_login", clear_on_submit=False):
            ugc_username = st.text_input("Username", 
                                       placeholder="Enter UGC/AICTE username",
                                       key="ugc_username")
            ugc_password = st.text_input("Password", 
                                       type="password", 
                                       placeholder="Enter password",
                                       key="ugc_password")
            
            # Additional security field
            with st.expander("Security Verification"):
                security_code = st.text_input("Security Code", 
                                            placeholder="Enter security code if required",
                                            type="password",
                                            key="security_code")
            
            ugc_submit = st.form_submit_button("✅ Login as UGC/AICTE")
            
            if ugc_submit:
                # Perform validation
                validation_passed = True
                
                # Validate username
                username_errors = validate_username(ugc_username, "ugc_aicte")
                if username_errors:
                    display_validation_errors(username_errors)
                    validation_passed = False
                
                # Validate password
                password_errors, password_warnings = validate_password(ugc_password, "ugc_aicte")
                if password_errors:
                    display_validation_errors(password_errors)
                    validation_passed = False
                if password_warnings:
                    display_validation_errors([], password_warnings)
                
                if validation_passed:
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
    <p>SUGAM Platform v2.0 | Access restricted to authorized personnel | Data validation enabled | {datetime.now().strftime("%d %B %Y")}</p>
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
    
    # User profile section with validation
    with st.sidebar.expander("👤 User Profile"):
        profile_form = st.form("user_profile_form")
        with profile_form:
            new_email = st.text_input("Update Email", 
                                    placeholder="Enter new email",
                                    key="profile_email")
            new_phone = st.text_input("Update Phone", 
                                    placeholder="Enter phone number",
                                    key="profile_phone")
            update_profile = st.form_submit_button("Update Profile")
            
            if update_profile:
                validation_passed = True
                
                if new_email:
                    email_errors = validate_email(new_email)
                    if email_errors:
                        display_validation_errors(email_errors)
                        validation_passed = False
                
                if new_phone:
                    phone_errors = validate_phone(new_phone)
                    if phone_errors:
                        display_validation_errors(phone_errors)
                        validation_passed = False
                
                if validation_passed:
                    st.success("✅ Profile updated successfully!")
    
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
            "🌳 Decision Tree Classifier",
            "🔮 Performance Prediction (Logistic Regression)",
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

    elif app_mode == "🌳 Decision Tree Classifier":
        create_decision_tree_module(analyzer)
        
        # Display system information
        st.subheader("System Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Records", len(analyzer.historical_data))
        with
