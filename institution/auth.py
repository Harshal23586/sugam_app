import streamlit as st

def create_institution_login(analyzer):
    st.header("🏛️ Institution Portal Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Existing Institution Users")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = analyzer.authenticate_institution_user(username, password)
            if user:
                st.session_state.institution_user = user
                st.session_state.user_role = "Institution"
                st.success(f"Welcome, {user['contact_person']} from {user['institution_name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.subheader("New Institution Registration")
        
        # Get available institutions
        available_institutions = analyzer.historical_data[
            analyzer.historical_data['year'] == 2023
        ][['institution_id', 'institution_name']].drop_duplicates()
        
        if not available_institutions.empty:
            selected_institution = st.selectbox(
                "Select Your Institution",
                available_institutions['institution_id'].tolist(),
                format_func=lambda x: available_institutions[
                    available_institutions['institution_id'] == x
                ]['institution_name'].iloc[0]
            )
            
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            contact_person = st.text_input("Contact Person Name")
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")
            
            if st.button("Register Institution Account"):
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif not all([new_username, new_password, contact_person, email]):
                    st.error("Please fill all required fields!")
                else:
                    success = analyzer.create_institution_user(
                        selected_institution, new_username, new_password,
                        contact_person, email, phone
                    )
                    if success:
                        st.success("Institution account created successfully! You can now login.")
                    else:
                        st.error("Username already exists. Please choose a different username.")
        else:
            st.warning("No institutions available for registration")
