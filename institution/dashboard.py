# institution/dashboard.py
"""
Institution Dashboard Module

This module provides the main dashboard interface for institutional users,
including document upload, data submission, submission tracking, and 
approval workflow management.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Any

# Import from project modules
from core.analyzer import InstitutionalAIAnalyzer
from institution.forms import (
    create_institution_data_submission,
    create_systematic_data_submission_form
)
from institution.documents import create_institution_document_upload
from institution.submissions import (
    create_institution_submissions_view,
    create_institution_requirements_guide,
    create_institution_approval_workflow
)

def sfr_drilldown(df):

    # Rename columns to safe internal names
    df = df.rename(columns={
        "Year": "year",
        "Type of Institute": "type",
        "Institute Code": "institute",
        "Student Faculty Ratio": "sfr"
    })

    # Initialize drill state
    if "sfr_drill_level" not in st.session_state:
        st.session_state.sfr_drill_level = 0
        st.session_state.sfr_drill_path = []

    # Drill structure based on your schema
    levels = [
        ("Institute Type", ["type"]),
        ("Institute Code", ["institute"]),
        ("Year", ["year"])
    ]

    level_label, group_cols = levels[st.session_state.sfr_drill_level]

    st.subheader(f"Student–Faculty Ratio Drilldown — {level_label}")

    # Breadcrumbs
    if st.session_state.sfr_drill_path:
        breadcrumb = " > ".join([str(x) for x in st.session_state.sfr_drill_path])
        st.markdown(f"**Path:** {breadcrumb}")
    else:
        st.markdown("**Path:** Top Level")

    # Back + Reset buttons
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅ Back") and st.session_state.sfr_drill_level > 0:
            st.session_state.sfr_drill_level -= 1
            st.session_state.sfr_drill_path.pop()
            st.experimental_rerun()

    with c2:
        if st.button("Reset"):
            st.session_state.sfr_drill_level = 0
            st.session_state.sfr_drill_path = []
            st.experimental_rerun()

    # Apply filters from drill path
    filtered = df.copy()
    for i, selected_val in enumerate(st.session_state.sfr_drill_path):
        filter_col = levels[i][1][0]
        filtered = filtered[filter_col] == selected_val
        df = df[df[filter_col] == selected_val]

    # Aggregate (mean SFR)
    agg = df.groupby(group_cols).agg(
        avg_sfr=('sfr', 'mean')
    ).reset_index()

    # Bar chart
    x_col = group_cols[0]
    st.bar_chart(agg.set_index(x_col)['avg_sfr'])

    # Drill-down selectbox
    options = ["-- None --"] + agg[x_col].astype(str).tolist()
    choice = st.selectbox(f"Select {level_label} to drill into:", options)

    if choice != "-- None --" and st.button("Drill"):
        st.session_state.sfr_drill_path.append(choice)
        if st.session_state.sfr_drill_level < len(levels) - 1:
            st.session_state.sfr_drill_level += 1
        st.experimental_rerun()

    # Display table under the chart
    st.dataframe(agg)

def create_institution_dashboard(analyzer: InstitutionalAIAnalyzer, user: Dict):
    """
    Main institution dashboard function
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: Dictionary containing user information
    """
    if not user:
        st.error("❌ No user data available. Please log in again.")
        return
    
    st.header(f"🏛️ Institution Dashboard - {user.get('institution_name', 'Unknown')}")
    
    # Display institution overview in a prominent card
    with st.container():
        st.markdown("### 👤 Institution Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Institution ID", user.get('institution_id', 'N/A'))
        with col2:
            st.metric("Contact Person", user.get('contact_person', 'N/A'))
        with col3:
            st.metric("Email", user.get('email', 'N/A'))
        with col4:
            st.metric("Role", user.get('role', 'Institution'))
        
        # Add a separator
        st.markdown("---")
    
    # Get institution's current performance data
    institution_performance = get_institution_performance_data(analyzer, user['institution_id'])
    
    # Display performance snapshot if available
    if institution_performance:
        display_performance_snapshot(institution_performance)
    
    # Navigation tabs for institution users
    st.markdown("### 📋 Dashboard Navigation")
    
    institution_tabs = st.tabs([
        "📤 Document Upload", 
        "📝 Basic Data Submission",
        "🏛️ Systematic Data Form",   
        "📊 My Submissions",
        "📋 Requirements Guide",
        "🔄 Approval Workflow",
        "📈 Performance Insights",
        "📊 SFR Drill Down"
    ])
    
    # Tab 1: Document Upload
    with institution_tabs[0]:
        create_institution_document_upload(analyzer, user)
    
    # Tab 2: Basic Data Submission
    with institution_tabs[1]:
        create_institution_data_submission(analyzer, user)
    
    # Tab 3: Systematic Data Form
    with institution_tabs[2]:
        create_systematic_data_submission_form(analyzer, user)
    
    # Tab 4: My Submissions
    with institution_tabs[3]:
        create_institution_submissions_view(analyzer, user)
    
    # Tab 5: Requirements Guide
    with institution_tabs[4]:
        create_institution_requirements_guide(analyzer)
    
    # Tab 6: Approval Workflow
    with institution_tabs[5]:
        create_institution_approval_workflow(analyzer, user)
    
    # Tab 7: Performance Insights (New)
    with institution_tabs[6]:
        create_performance_insights(analyzer, user, institution_performance)

    # Tab 8: SFR Drill Down (New)
    with institution_tabs[6]:
        sfr_drilldown(analyzer.historical_data)
        

def get_institution_performance_data(analyzer: InstitutionalAIAnalyzer, institution_id: str) -> Optional[Dict]:
    """
    Get current performance data for the institution
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        institution_id: ID of the institution
    
    Returns:
        Dictionary with performance data or None if not found
    """
    try:
        # Get current year data
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        
        # Find institution data
        institution_data = current_year_data[current_year_data['institution_id'] == institution_id]
        
        if institution_data.empty:
            # Try to get any year data
            all_data = analyzer.historical_data[analyzer.historical_data['institution_id'] == institution_id]
            if not all_data.empty:
                institution_data = all_data[all_data['year'] == all_data['year'].max()]
        
        if not institution_data.empty:
            return institution_data.iloc[0].to_dict()
        
    except Exception as e:
        st.error(f"Error fetching performance data: {str(e)}")
    
    return None

def display_performance_snapshot(performance_data: Dict):
    """
    Display a snapshot of institution's performance
    
    Args:
        performance_data: Dictionary containing performance metrics
    """
    st.subheader("📊 Performance Snapshot")
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = performance_data.get('performance_score', 0)
        # Color code based on performance
        if score >= 8.0:
            st.success(f"**Performance Score**: {score:.1f}/10")
        elif score >= 6.0:
            st.info(f"**Performance Score**: {score:.1f}/10")
        else:
            st.warning(f"**Performance Score**: {score:.1f}/10")
    
    with col2:
        risk_level = performance_data.get('risk_level', 'Unknown')
        if risk_level == "Low Risk":
            st.success(f"**Risk Level**: {risk_level}")
        elif risk_level == "Medium Risk":
            st.info(f"**Risk Level**: {risk_level}")
        else:
            st.error(f"**Risk Level**: {risk_level}")
    
    with col3:
        naac_grade = performance_data.get('naac_grade', 'Not Rated')
        if naac_grade in ['A++', 'A+', 'A']:
            st.success(f"**NAAC Grade**: {naac_grade}")
        elif naac_grade in ['B++', 'B+']:
            st.info(f"**NAAC Grade**: {naac_grade}")
        else:
            st.warning(f"**NAAC Grade**: {naac_grade}")
    
    with col4:
        placement_rate = performance_data.get('placement_rate', 0)
        if placement_rate >= 80:
            st.success(f"**Placement Rate**: {placement_rate:.1f}%")
        elif placement_rate >= 60:
            st.info(f"**Placement Rate**: {placement_rate:.1f}%")
        else:
            st.warning(f"**Placement Rate**: {placement_rate:.1f}%")
    
    # Display approval status
    approval_status = performance_data.get('approval_recommendation', 'Under Review')
    with st.container():
        st.markdown("#### 📋 Approval Status")
        
        if "Full Approval" in approval_status:
            st.success(f"✅ **{approval_status}**")
        elif "Provisional" in approval_status or "Conditional" in approval_status:
            st.info(f"⚠️ **{approval_status}**")
        elif "Rejection" in approval_status:
            st.error(f"❌ **{approval_status}**")
        else:
            st.info(f"📝 **{approval_status}**")
    
    # Quick action buttons
    st.markdown("#### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.active_tab = "📤 Document Upload"
            st.rerun()
    
    with col2:
        if st.button("📝 Submit Data", use_container_width=True):
            st.session_state.active_tab = "📝 Basic Data Submission"
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.active_tab = "📈 Performance Insights"
            st.rerun()

def create_performance_insights(analyzer: InstitutionalAIAnalyzer, user: Dict, performance_data: Optional[Dict] = None):
    """
    Create performance insights and analytics for the institution
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
        user: Dictionary containing user information
        performance_data: Current performance data (optional)
    """
    st.subheader("📈 Performance Insights & Analytics")
    
    if not performance_data:
        st.info("No performance data available. Please submit your institutional data first.")
        return
    
    # Get historical data for trend analysis
    historical_data = analyzer.historical_data[
        analyzer.historical_data['institution_id'] == user['institution_id']
    ].sort_values('year')
    
    # Create metrics dashboard
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = performance_data.get('performance_score', 0)
        st.metric("Performance Score", f"{score:.2f}/10")
    
    with col2:
        placement = performance_data.get('placement_rate', 0)
        st.metric("Placement Rate", f"{placement:.1f}%")
    
    with col3:
        research = performance_data.get('research_publications', 0)
        st.metric("Research Output", research)
    
    with col4:
        sf_ratio = performance_data.get('student_faculty_ratio', 0)
        st.metric("Student-Faculty Ratio", f"{sf_ratio:.1f}:1")
    
    # Trend Analysis
    if len(historical_data) > 1:
        st.markdown("### 📈 Performance Trend")
        
        fig = px.line(
            historical_data,
            x='year',
            y='performance_score',
            markers=True,
            title="Performance Score Trend Over Time",
            labels={'performance_score': 'Performance Score', 'year': 'Year'}
        )
        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(range=[0, 10])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparative Analysis
    st.markdown("### 🏛️ Peer Comparison")
    
    # Get peer institutions (same type)
    current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    peer_institutions = current_year_data[
        (current_year_data['institution_type'] == performance_data.get('institution_type')) &
        (current_year_data['institution_id'] != user['institution_id'])
    ]
    
    if not peer_institutions.empty:
        # Calculate percentile
        all_scores = peer_institutions['performance_score'].tolist() + [performance_data.get('performance_score', 0)]
        percentile = (sum(1 for s in all_scores if s < performance_data.get('performance_score', 0)) / len(all_scores)) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Performance Percentile", f"{percentile:.1f}%")
        
        with col2:
            peer_avg = peer_institutions['performance_score'].mean()
            diff = performance_data.get('performance_score', 0) - peer_avg
            st.metric("vs Peer Average", f"{diff:+.2f}", delta=f"{diff:+.2f}")
        
        # Show top 5 peers
        st.markdown("##### Top 5 Peer Institutions")
        top_peers = peer_institutions.nlargest(5, 'performance_score')[['institution_name', 'performance_score', 'naac_grade']]
        st.dataframe(top_peers, use_container_width=True)
    
    # Strengths and Weaknesses Analysis
    st.markdown("### 🎯 Strengths & Areas for Improvement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths")
        strengths = identify_strengths(performance_data)
        if strengths:
            for strength in strengths[:5]:  # Show top 5 strengths
                st.success(f"✓ {strength}")
        else:
            st.info("No significant strengths identified")
    
    with col2:
        st.markdown("#### 📝 Areas for Improvement")
        weaknesses = identify_weaknesses(performance_data)
        if weaknesses:
            for weakness in weaknesses[:5]:  # Show top 5 weaknesses
                st.error(f"⚠ {weakness}")
        else:
            st.info("No major areas for improvement identified")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    recommendations = generate_recommendations(performance_data)
    
    for i, rec in enumerate(recommendations[:5], 1):
        st.info(f"{i}. {rec}")
    
    # Download insights report
    if st.button("📥 Download Insights Report"):
        generate_insights_report(user, performance_data, historical_data)

def identify_strengths(performance_data: Dict) -> List[str]:
    """
    Identify institutional strengths based on performance data
    
    Args:
        performance_data: Dictionary containing performance metrics
    
    Returns:
        List of strength statements
    """
    strengths = []
    
    # NAAC Grade
    naac_grade = performance_data.get('naac_grade')
    if naac_grade in ['A++', 'A+', 'A']:
        strengths.append(f"Strong NAAC Accreditation ({naac_grade})")
    
    # Performance Score
    performance_score = performance_data.get('performance_score', 0)
    if performance_score >= 8.0:
        strengths.append(f"High Overall Performance ({performance_score:.1f}/10)")
    
    # Placement Rate
    placement_rate = performance_data.get('placement_rate', 0)
    if placement_rate >= 80:
        strengths.append(f"Excellent Placement Record ({placement_rate:.1f}%)")
    
    # Research Output
    research_publications = performance_data.get('research_publications', 0)
    if research_publications >= 50:
        strengths.append(f"Robust Research Output ({research_publications} publications)")
    
    # Financial Stability
    financial_stability = performance_data.get('financial_stability_score', 0)
    if financial_stability >= 8.0:
        strengths.append(f"Strong Financial Stability ({financial_stability:.1f}/10)")
    
    # Student-Faculty Ratio
    sf_ratio = performance_data.get('student_faculty_ratio', 0)
    if sf_ratio <= 15:
        strengths.append(f"Favorable Student-Faculty Ratio ({sf_ratio:.1f}:1)")
    
    return strengths

def identify_weaknesses(performance_data: Dict) -> List[str]:
    """
    Identify institutional weaknesses based on performance data
    
    Args:
        performance_data: Dictionary containing performance metrics
    
    Returns:
        List of weakness statements
    """
    weaknesses = []
    
    # Performance Score
    performance_score = performance_data.get('performance_score', 0)
    if performance_score < 6.0:
        weaknesses.append(f"Low Overall Performance ({performance_score:.1f}/10)")
    
    # Placement Rate
    placement_rate = performance_data.get('placement_rate', 0)
    if placement_rate < 65:
        weaknesses.append(f"Low Placement Rate ({placement_rate:.1f}%)")
    
    # Student-Faculty Ratio
    sf_ratio = performance_data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        weaknesses.append(f"High Student-Faculty Ratio ({sf_ratio:.1f}:1)")
    
    # Research Output
    research_publications = performance_data.get('research_publications', 0)
    if research_publications < 20:
        weaknesses.append(f"Limited Research Output ({research_publications} publications)")
    
    # Digital Infrastructure
    digital_score = performance_data.get('digital_infrastructure_score', 0)
    if digital_score < 6.0:
        weaknesses.append(f"Weak Digital Infrastructure ({digital_score:.1f}/10)")
    
    # Community Engagement
    community_projects = performance_data.get('community_projects', 0)
    if community_projects < 5:
        weaknesses.append("Limited Community Engagement")
    
    return weaknesses

def generate_recommendations(performance_data: Dict) -> List[str]:
    """
    Generate improvement recommendations based on performance data
    
    Args:
        performance_data: Dictionary containing performance metrics
    
    Returns:
        List of recommendation statements
    """
    recommendations = []
    
    # Student-Faculty Ratio
    sf_ratio = performance_data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        recommendations.append("Recruit additional faculty members to improve student-faculty ratio")
    
    # Placement Rate
    placement_rate = performance_data.get('placement_rate', 0)
    if placement_rate < 70:
        recommendations.append("Strengthen industry partnerships and career development programs")
    
    # Research Output
    research_publications = performance_data.get('research_publications', 0)
    if research_publications < 30:
        recommendations.append("Establish research promotion policy and faculty development programs")
    
    # Digital Infrastructure
    digital_score = performance_data.get('digital_infrastructure_score', 0)
    if digital_score < 7:
        recommendations.append("Invest in digital infrastructure and e-learning platforms")
    
    # Community Engagement
    community_projects = performance_data.get('community_projects', 0)
    if community_projects < 5:
        recommendations.append("Enhance community engagement and social outreach programs")
    
    # NIRF Participation
    if not performance_data.get('nirf_ranking'):
        recommendations.append("Participate in NIRF ranking to enhance institutional reputation")
    
    # General recommendations
    recommendations.append("Participate in government schemes like RUSA, FIST, NMEICT")
    recommendations.append("Explore international collaborations and student exchange programs")
    
    return recommendations

def generate_insights_report(user: Dict, performance_data: Dict, historical_data: pd.DataFrame):
    """
    Generate and download insights report
    
    Args:
        user: User information
        performance_data: Current performance data
        historical_data: Historical performance data
    """
    # Create report data
    report = {
        "report_type": "Institutional Insights Report",
        "generated_date": datetime.now().isoformat(),
        "institution_info": {
            "name": user.get('institution_name'),
            "id": user.get('institution_id'),
            "contact": user.get('contact_person'),
            "email": user.get('email')
        },
        "performance_summary": {
            "current_score": performance_data.get('performance_score'),
            "risk_level": performance_data.get('risk_level'),
            "approval_status": performance_data.get('approval_recommendation'),
            "naac_grade": performance_data.get('naac_grade')
        },
        "key_metrics": {
            "placement_rate": performance_data.get('placement_rate'),
            "research_publications": performance_data.get('research_publications'),
            "student_faculty_ratio": performance_data.get('student_faculty_ratio'),
            "financial_stability": performance_data.get('financial_stability_score')
        },
        "strengths": identify_strengths(performance_data)[:5],
        "improvement_areas": identify_weaknesses(performance_data)[:5],
        "recommendations": generate_recommendations(performance_data)[:5]
    }
    
    # Convert to JSON for download
    report_json = json.dumps(report, indent=2)
    
    # Create download button
    st.download_button(
        label="📥 Download Insights Report (JSON)",
        data=report_json,
        file_name=f"insights_report_{user.get('institution_id')}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

# Main execution guard for testing
if __name__ == "__main__":
    # This allows testing the module directly
    st.set_page_config(page_title="Institution Dashboard Test", layout="wide")
    
    st.title("Institution Dashboard Test")
    
    # Create a dummy user for testing
    dummy_user = {
        'institution_id': 'INST_0001',
        'institution_name': 'Test University',
        'contact_person': 'Dr. Test User',
        'email': 'test@university.edu',
        'role': 'Institution'
    }
    
    # Initialize analyzer
    analyzer = InstitutionalAIAnalyzer()
    
    # Create dashboard
    create_institution_dashboard(analyzer, dummy_user)








