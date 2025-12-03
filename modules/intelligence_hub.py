import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def create_institutional_intelligence_hub(analyzer):
    st.header("🧠 Institutional Intelligence Hub")
    
    st.info("""
    **Comprehensive AI-powered insights, predictive analytics, and strategic recommendations** 
    for institutional excellence and NEP 2020 compliance.
    """)
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
    
    selected_institution = st.selectbox(
        "Select Institution for Deep Analysis",
        current_institutions,
        key="intel_hub_institution"
    )
    
    if selected_institution:
        # Get institution data
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        inst_data = current_year_data[current_year_data['institution_id'] == selected_institution]
        
        if not inst_data.empty:
            display_intelligence_dashboard(inst_data.iloc[0], analyzer)
        else:
            st.warning("No data found for selected institution")

def display_intelligence_dashboard(institution_data, analyzer):
    """Display comprehensive intelligence dashboard"""
    
    st.subheader(f"🏛️ {institution_data['institution_name']}")
    
    # Executive Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        performance_score = institution_data.get('performance_score', 0)
        if performance_score >= 8.0:
            st.success(f"**Performance**: {performance_score:.1f}/10")
        elif performance_score >= 6.0:
            st.info(f"**Performance**: {performance_score:.1f}/10")
        else:
            st.warning(f"**Performance**: {performance_score:.1f}/10")
    
    with col2:
        risk_level = institution_data.get('risk_level', 'Medium Risk')
        if risk_level == "Low Risk":
            st.success(f"**Risk**: {risk_level}")
        elif risk_level == "Medium Risk":
            st.info(f"**Risk**: {risk_level}")
        else:
            st.error(f"**Risk**: {risk_level}")
    
    with col3:
        approval_status = institution_data.get('approval_recommendation', 'Under Review')
        st.metric("Approval Status", approval_status)
    
    with col4:
        st.metric("Institution Type", institution_data.get('institution_type', 'N/A'))
    
    # Key Metrics
    st.subheader("📊 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        placement = institution_data.get('placement_rate', 0)
        st.metric("Placement Rate", f"{placement:.1f}%")
    
    with kpi_col2:
        research = institution_data.get('research_publications', 0)
        st.metric("Research Papers", research)
    
    with kpi_col3:
        sf_ratio = institution_data.get('student_faculty_ratio', 0)
        st.metric("Student-Faculty Ratio", f"{sf_ratio:.1f}")
    
    with kpi_col4:
        naac = institution_data.get('naac_grade', 'N/A')
        st.metric("NAAC Grade", naac)
    
    # AI Insights
    st.subheader("🤖 AI-Powered Insights")
    
    tab1, tab2, tab3 = st.tabs(["Strengths", "Areas for Improvement", "Recommendations"])
    
    with tab1:
        strengths = get_institutional_strengths(institution_data)
        for strength in strengths:
            st.success(f"✅ {strength}")
    
    with tab2:
        weaknesses = get_institutional_weaknesses(institution_data)
        for weakness in weaknesses:
            st.error(f"❌ {weakness}")
    
    with tab3:
        recommendations = get_ai_recommendations(institution_data)
        for rec in recommendations:
            st.info(f"💡 {rec}")

def get_ai_recommendations(data):
    """Get AI recommendations"""
    recommendations = []
    
    sf_ratio = data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        recommendations.append("Recruit additional faculty to improve student-faculty ratio")
    
    placement = data.get('placement_rate', 0)
    if placement < 70:
        recommendations.append("Strengthen industry partnerships and career services")
    
    research = data.get('research_publications', 0)
    if research < 30:
        recommendations.append("Establish research promotion programs and incentives")
    
    if not data.get('nirf_ranking'):
        recommendations.append("Participate in NIRF ranking for better visibility")
    
    return recommendations if recommendations else ["Institution is performing well across parameters"]

def get_institutional_weaknesses(data):
    """Get institutional weaknesses"""
    weaknesses = []
    
    sf_ratio = data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        weaknesses.append(f"High student-faculty ratio: {sf_ratio:.1f}")
    
    placement = data.get('placement_rate', 0)
    if placement < 65:
        weaknesses.append(f"Low placement rate: {placement:.1f}%")
    
    research = data.get('research_publications', 0)
    if research < 20:
        weaknesses.append(f"Limited research output: {research} publications")
    
    return weaknesses if weaknesses else ["No major weaknesses identified"]

def get_institutional_strengths(data):
    """Get institutional strengths"""
    strengths = []
    
    naac = data.get('naac_grade', '')
    if naac in ['A++', 'A+', 'A']:
        strengths.append(f"Strong NAAC accreditation: {naac}")
    
    placement = data.get('placement_rate', 0)
    if placement > 80:
        strengths.append(f"Excellent placement rate: {placement:.1f}%")
    
    research = data.get('research_publications', 0)
    if research > 50:
        strengths.append(f"Robust research output: {research} publications")
    
    return strengths if strengths else ["No significant strengths identified"]
