import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
        # Get comprehensive data
        institution_data = get_comprehensive_institution_data(selected_institution, analyzer)
        
        if institution_data:
            display_intelligence_dashboard(institution_data, analyzer)
        
        # Comparative analysis section
        st.subheader("🏛️ Peer Comparison")
        
        if st.button("📊 Generate Comparative Analysis", type="secondary"):
            show_comparative_analysis_safe(selected_institution, analyzer)
        
        # Improvement roadmap
        st.subheader("🎯 Improvement Strategy")
        
        if st.button("🔄 Generate Improvement Roadmap", type="secondary"):
            show_improvement_roadmap_safe(selected_institution, analyzer)

def get_comprehensive_institution_data(institution_id, analyzer):
    """Get comprehensive data for intelligence analysis"""
    try:
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        current_data = current_year_data[current_year_data['institution_id'] == institution_id]
        
        if current_data.empty:
            return None
        
        historical_data = analyzer.historical_data[
            analyzer.historical_data['institution_id'] == institution_id
        ].sort_values('year')
        
        return {
            'current': current_data.iloc[0].to_dict(),
            'historical': historical_data,
            'institution_id': institution_id
        }
        
    except Exception as e:
        st.error(f"Error loading institution data: {str(e)}")
        return None

def display_intelligence_dashboard(data, analyzer):
    """Display comprehensive intelligence dashboard"""
    
    current = data['current']
    historical = data['historical']
    
    # Executive Summary
    st.subheader("🎯 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        performance_score = current.get('performance_score', 0)
        if performance_score >= 8.0:
            st.success(f"**Performance**: {performance_score:.1f}/10")
        elif performance_score >= 6.0:
            st.info(f"**Performance**: {performance_score:.1f}/10")
        else:
            st.warning(f"**Performance**: {performance_score:.1f}/10")
    
    with col2:
        risk_level = current.get('risk_level', 'Medium Risk')
        if risk_level == "Low Risk":
            st.success(f"**Risk**: {risk_level}")
        elif risk_level == "Medium Risk":
            st.info(f"**Risk**: {risk_level}")
        else:
            st.error(f"**Risk**: {risk_level}")
    
    with col3:
        approval_status = current.get('approval_recommendation', 'Under Review')
        st.metric("Approval Status", approval_status)
    
    with col4:
        naac_grade = current.get('naac_grade', 'N/A')
        st.metric("NAAC Grade", naac_grade)
    
    # Performance Trend
    st.subheader("📈 Performance Trend (2014-2023)")
    
    if len(historical) > 1:
        fig = px.line(
            historical,
            x='year',
            y='performance_score',
            markers=True,
            title=f"Performance Trend: {historical['year'].min()}-{historical['year'].max()}",
            line_shape='linear'
        )
        
        # Add trend line
        z = np.polyfit(range(len(historical)), historical['performance_score'], 1)
        p = np.poly1d(z)
        
        fig.add_scatter(
            x=historical['year'],
            y=p(range(len(historical))),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics Grid
    st.subheader("📊 Performance Metrics Dashboard")
    
    metrics_grid = st.columns(4)
    
    metrics = [
        ("🎓 Placement Rate", current.get('placement_rate', 0), f"{current.get('placement_rate', 0):.1f}%"),
        ("🔬 Research Papers", current.get('research_publications', 0), current.get('research_publications', 0)),
        ("👨‍🏫 S/F Ratio", current.get('student_faculty_ratio', 0), f"{current.get('student_faculty_ratio', 0):.1f}"),
        ("💰 Financial Stability", current.get('financial_stability_score', 0), f"{current.get('financial_stability_score', 0):.1f}/10"),
    ]
    
    for i, (title, value, display) in enumerate(metrics):
        with metrics_grid[i]:
            st.metric(title, display)
    
    # Performance Radar Chart
    st.subheader("📊 Performance Radar Chart")
    
    categories = [
        'Academic Excellence',
        'Research & Innovation', 
        'Infrastructure',
        'Governance',
        'Student Development',
        'Social Impact'
    ]
    
    # Simulated scores for each category
    scores = [
        min(10, performance_score * 1.1),
        min(10, current.get('research_publications', 0) / 10),
        current.get('digital_infrastructure_score', 7),
        current.get('financial_stability_score', 7.5),
        current.get('placement_rate', 75) / 10,
        min(10, current.get('community_projects', 0) * 0.5)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Performance Across Categories"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    st.subheader("🤖 AI-Powered Insights")
    
    insights_tabs = st.tabs(["Strengths", "Weaknesses", "Opportunities", "Threats"])
    
    with insights_tabs[0]:
        strengths = identify_institutional_strengths(current)
        if strengths:
            for strength in strengths:
                st.success(f"✅ {strength}")
        else:
            st.info("No significant strengths identified")
    
    with insights_tabs[1]:
        weaknesses = identify_institutional_weaknesses(current)
        if weaknesses:
            for weakness in weaknesses:
                st.error(f"❌ {weakness}")
        else:
            st.info("No major weaknesses identified")
    
    with insights_tabs[2]:
        opportunities = identify_opportunities(current)
        if opportunities:
            for opportunity in opportunities:
                st.info(f"🔍 {opportunity}")
        else:
            st.info("No specific opportunities identified")
    
    with insights_tabs[3]:
        threats = [
            "Increasing competition from private institutions",
            "Changing regulatory requirements",
            "Funding constraints",
            "Faculty retention challenges"
        ]
        for threat in threats:
            st.warning(f"⚠️ {threat}")
    
    # Benchmarking
    st.subheader("📊 Benchmarking Against Peers")
    
    peer_data = analyzer.historical_data[
        (analyzer.historical_data['institution_type'] == current['institution_type']) &
        (analyzer.historical_data['year'] == 2023) &
        (analyzer.historical_data['institution_id'] != data['institution_id'])
    ]
    
    if not peer_data.empty:
        peer_avg = peer_data['performance_score'].mean()
        current_score = current['performance_score']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Your Score", f"{current_score:.1f}")
        
        with col2:
            st.metric("Peer Average", f"{peer_avg:.1f}")
        
        with col3:
            diff = current_score - peer_avg
            st.metric("Difference", f"{diff:+.1f}", delta=f"{diff:+.1f}")
        
        if diff > 0:
            st.success(f"🎯 Your institution performs **{diff:.1f} points better** than the peer average!")
        elif diff < 0:
            st.warning(f"📉 Your institution is **{abs(diff):.1f} points below** the peer average.")
        else:
            st.info("📊 Your institution performs at par with peers.")
    else:
        st.info("No peer institutions found for comparison")

def identify_institutional_strengths(data):
    """Identify institutional strengths"""
    strengths = []
    
    naac_grade = data.get('naac_grade', '')
    if naac_grade in ['A++', 'A+', 'A']:
        strengths.append(f"Strong NAAC Accreditation: {naac_grade}")
    
    placement_rate = data.get('placement_rate', 0)
    if placement_rate > 80:
        strengths.append(f"Excellent Placement Record: {placement_rate:.1f}%")
    
    research_pubs = data.get('research_publications', 0)
    if research_pubs > 50:
        strengths.append(f"Robust Research Output: {research_pubs} publications")
    
    financial_score = data.get('financial_stability_score', 0)
    if financial_score > 8.0:
        strengths.append(f"Excellent Financial Stability: {financial_score:.1f}/10")
    
    digital_score = data.get('digital_infrastructure_score', 0)
    if digital_score > 8.0:
        strengths.append(f"Advanced Digital Infrastructure: {digital_score:.1f}/10")
    
    return strengths[:5]

def identify_institutional_weaknesses(data):
    """Identify institutional weaknesses"""
    weaknesses = []
    
    sf_ratio = data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        weaknesses.append(f"High Student-Faculty Ratio: {sf_ratio:.1f}")
    
    placement_rate = data.get('placement_rate', 0)
    if placement_rate < 65:
        weaknesses.append(f"Low Placement Rate: {placement_rate:.1f}%")
    
    research_pubs = data.get('research_publications', 0)
    if research_pubs < 20:
        weaknesses.append(f"Inadequate Research Output: {research_pubs} publications")
    
    digital_score = data.get('digital_infrastructure_score', 0)
    if digital_score < 7:
        weaknesses.append(f"Weak Digital Infrastructure: {digital_score:.1f}/10")
    
    naac_grade = data.get('naac_grade', '')
    if naac_grade in ['C', 'B']:
        weaknesses.append(f"Below Average NAAC Grade: {naac_grade}")
    
    return weaknesses[:5]

def identify_opportunities(data):
    """Identify opportunities for improvement"""
    opportunities = []
    
    if not data.get('nirf_ranking'):
        opportunities.append("Participate in NIRF ranking for better visibility")
    
    industry_collabs = data.get('industry_collaborations', 0)
    if industry_collabs < 5:
        opportunities.append("Increase industry collaborations for practical exposure")
    
    patents = data.get('patents_filed', 0)
    if patents < 3:
        opportunities.append("Strengthen IPR culture and patent filings")
    
    digital_score = data.get('digital_infrastructure_score', 0)
    if digital_score < 8:
        opportunities.append("Invest in digital infrastructure for blended learning")
    
    opportunities.append("Explore international collaborations and student exchange programs")
    opportunities.append("Participate in government schemes like RUSA, FIST, NMEICT")
    
    return opportunities[:5]

def show_comparative_analysis_safe(institution_id, analyzer):
    """Safe version of comparative analysis"""
    try:
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        current_data = current_year_data[current_year_data['institution_id'] == institution_id]
        
        if current_data.empty:
            st.warning("No data available for current institution")
            return
        
        current_row = current_data.iloc[0]
        
        peer_data = current_year_data[
            (current_year_data['institution_type'] == current_row['institution_type']) &
            (current_year_data['institution_id'] != institution_id)
        ]
        
        if peer_data.empty:
            st.info("No peer institutions found for comparison")
            return
        
        comparison = [{
            'Institution': f"📍 {current_row['institution_name']}",
            'Type': current_row['institution_type'],
            'Performance': current_row['performance_score'],
            'NAAC Grade': current_row.get('naac_grade', 'N/A'),
            'Placement %': current_row.get('placement_rate', 0)
        }]
        
        top_peers = peer_data.nlargest(3, 'performance_score')
        for _, peer in top_peers.iterrows():
            comparison.append({
                'Institution': peer['institution_name'],
                'Type': peer['institution_type'],
                'Performance': peer['performance_score'],
                'NAAC Grade': peer.get('naac_grade', 'N/A'),
                'Placement %': peer.get('placement_rate', 0)
            })
        
        df_comparison = pd.DataFrame(comparison)
        st.dataframe(df_comparison, use_container_width=True)
        
        all_scores = peer_data['performance_score'].tolist() + [current_row['performance_score']]
        percentile = (sum(1 for s in all_scores if s < current_row['performance_score']) / len(all_scores)) * 100
        st.metric("Performance Percentile", f"{percentile:.1f}%")
        
    except Exception as e:
        st.error(f"Error in comparative analysis: {str(e)}")

def show_improvement_roadmap_safe(institution_id, analyzer):
    """Safe version of improvement roadmap"""
    try:
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        current_data = current_year_data[current_year_data['institution_id'] == institution_id]
        
        if current_data.empty:
            st.warning("No data available for institution")
            return
        
        current = current_data.iloc[0].to_dict()
        
        roadmap = [
            {"Priority": "High", "Action": "Review and update curriculum", "Timeline": "3 months", "Impact": "High"},
            {"Priority": "High", "Action": "Faculty development programs", "Timeline": "6 months", "Impact": "High"},
            {"Priority": "Medium", "Action": "Enhance research infrastructure", "Timeline": "9 months", "Impact": "Medium"},
            {"Priority": "Medium", "Action": "Improve digital learning facilities", "Timeline": "6 months", "Impact": "Medium"},
            {"Priority": "High", "Action": "Strengthen industry partnerships", "Timeline": "4 months", "Impact": "High"},
        ]
        
        weaknesses = identify_institutional_weaknesses(current)
        
        if "High student-faculty ratio" in str(weaknesses):
            roadmap.append({"Priority": "Critical", "Action": "Recruit additional faculty", "Timeline": "6 months", "Impact": "High"})
        
        if "Low placement rate" in str(weaknesses):
            roadmap.append({"Priority": "Critical", "Action": "Enhance placement services", "Timeline": "3 months", "Impact": "High"})
        
        df_roadmap = pd.DataFrame(roadmap)
        
        # Display roadmap with styling
        st.dataframe(
            df_roadmap.style.applymap(
                lambda x: 'background-color: #ffcccc' if x == 'Critical' 
                else 'background-color: #ffebcc' if x == 'High'
                else 'background-color: #e6f7ff' if x == 'Medium'
                else '',
                subset=['Priority']
            ),
            use_container_width=True
        )
        
        # Timeline visualization
        st.subheader("📅 Implementation Timeline")
        
        timeline_data = []
        for _, row in df_roadmap.iterrows():
            timeline_data.append({
                'Task': row['Action'],
                'Start': '2024-01-01',
                'Finish': f"2024-{int(row['Timeline'].split()[0])//3+1:02d}-01",
                'Priority': row['Priority']
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            # Simple bar chart for timeline
            fig = px.bar(
                timeline_df,
                x='Task',
                y=[1] * len(timeline_df),
                color='Priority',
                title="Implementation Timeline",
                color_discrete_map={
                    'Critical': '#ff4444',
                    'High': '#ffaa44',
                    'Medium': '#44aaff'
                }
            )
            fig.update_layout(showlegend=True, yaxis_title="", yaxis_showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating roadmap: {str(e)}")
