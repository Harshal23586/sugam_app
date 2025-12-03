import streamlit as st
import pandas as pd
import plotly.express as px

def create_data_management_module(analyzer):
    st.header("💾 Data Management & Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Current Data Analytics",
        "🔍 Data Validation & QA",
        "⚙️ Advanced Database Tools"
    ])
    
    with tab1:
        st.subheader("📊 Current Database Analytics")
        
        # Use the actual data from analyzer
        current_data = analyzer.historical_data
        
        # Show database statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_records = len(current_data)
            st.metric("📊 Total Records", total_records)
        
        with col2:
            unique_institutions = current_data['institution_id'].nunique()
            st.metric("🏛️ Unique Institutions", unique_institutions)
        
        with col3:
            years_covered = current_data['year'].nunique()
            st.metric("📅 Years Covered", years_covered)
        
        with col4:
            year_range = f"{current_data['year'].min()}-{current_data['year'].max()}"
            st.metric("🗓️ Year Range", year_range)
        
        # Show current year data
        current_year_data = current_data[current_data['year'] == 2023]
        
        st.subheader("📋 Data Preview (Current Year)")
        st.dataframe(current_year_data.head(), use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Data Quality Analysis & Validation")
        st.info("Data validation tools will be implemented here")
    
    with tab3:
        st.subheader("⚙️ Advanced Database Management")
        st.info("Database management tools will be implemented here")
