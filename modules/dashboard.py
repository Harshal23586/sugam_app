import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def sfr_drilldown(df):

    # rename columns to consistent internal names
    df = df.rename(columns={
        "year": "year",
        "institution_type": "type",
        "institution_id": "institute",
        "student_faculty_ratio": "sfr"
    })

    # initial state
    if "sfr_drill_level" not in st.session_state:
        st.session_state.sfr_drill_level = 0
        st.session_state.sfr_drill_path = []

    # new drill order: Year → Type → Institute
    levels = [
        ("Year", ["year"]),
        ("Institute Type", ["type"]),
        ("Institute Code", ["institute"])
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
            st.rerun()

    with c2:
        if st.button("Reset"):
            st.session_state.sfr_drill_level = 0
            st.session_state.sfr_drill_path = []
            st.rerun()

    # Apply filters from drill path
    filtered = df.copy()

    for i, selected_val in enumerate(st.session_state.sfr_drill_path):
        filter_col = levels[i][1][0]
        filtered = filtered[filtered[filter_col] == selected_val]
        df = df[df[filter_col] == selected_val]

    df = filtered

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
        st.rerun()

    # Display table under the chart
    st.dataframe(agg)

def create_performance_dashboard(analyzer):
    st.header("📊 Institutional Performance Analytics Dashboard")
    
    df = analyzer.historical_data
    
    # Show data specification
    st.info(f"📊 **Data Overview**: {df['institution_id'].nunique()} Institutions × {df['year'].nunique()} Years ({df['year'].min()}-{df['year'].max()}) | Total Records: {len(df)}")
    
    current_year_data = df[df['year'] == 2023]
    
    if len(current_year_data) == 0:
        st.warning("No data available for the current year.")
        return
    
    # Key Performance Indicators
    st.subheader("🏆 Key Performance Indicators (2023 Data)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_performance = current_year_data['performance_score'].mean()
        st.metric("Average Performance Score", f"{avg_performance:.2f}/10")
    
    with col2:
        approval_rate = (current_year_data['performance_score'] >= 6.0).mean()
        st.metric("Approval Eligibility Rate", f"{approval_rate:.1%}")
    
    with col3:
        high_risk_count = (current_year_data['risk_level'] == 'High Risk').sum() + (
            current_year_data['risk_level'] == 'Critical Risk').sum()
        st.metric("High/Critical Risk Institutions", high_risk_count)
    
    with col4:
        avg_placement = current_year_data['placement_rate'].mean()
        st.metric("Average Placement Rate", f"{avg_placement:.1f}%")
    
    with col5:
        research_intensity = current_year_data['research_publications'].sum() / len(current_year_data)
        st.metric("Avg Research Publications", f"{research_intensity:.1f}")
    
    # Performance Distribution
    st.subheader("📈 Performance Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            current_year_data, 
            x='performance_score',
            title="Distribution of Performance Scores",
            nbins=12,
            color_discrete_sequence=['#1f77b4'],
            opacity=0.8
        )
        fig1.update_layout(xaxis_title="Performance Score", yaxis_title="Count")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.box(
            current_year_data,
            x='institution_type',
            y='performance_score',
            title="Performance by Institution Type",
            color='institution_type'
        )
        fig2.update_layout(xaxis_title="Institution Type", yaxis_title="Performance Score")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Historical Trends
    st.subheader("📅 Historical Performance Trends")
    
    trend_data = df.groupby(['year', 'institution_type'])['performance_score'].mean().reset_index()
    
    fig3 = px.line(
        trend_data,
        x='year',
        y='performance_score',
        color='institution_type',
        title="Average Performance Score Trend (2014-2023)",
        markers=True
    )
    fig3.update_layout(xaxis_title="Year", yaxis_title="Average Performance Score")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_distribution = current_year_data['risk_level'].value_counts()
        fig4 = px.pie(
            values=risk_distribution.values,
            names=risk_distribution.index,
            title="Risk Level Distribution",
            color=risk_distribution.index,
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c',
                'Critical Risk': '#c0392b'
            }
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        fig5 = px.scatter(
            current_year_data,
            x='research_publications',
            y='placement_rate',
            color='risk_level',
            size='performance_score',
            hover_data=['institution_name'],
            title="Research vs Placement Analysis",
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c',
                'Critical Risk': '#c0392b'
            }
        )
        fig5.update_layout(xaxis_title="Research Publications", yaxis_title="Placement Rate (%)")
        st.plotly_chart(fig5, use_container_width=True)
    
    # Top Performers
    st.subheader("🏆 Student–Faculty Ratio Drilldown")
    sfr_drilldown(analyzer.historical_data)

    # Top Performers
    st.subheader("🏆 Top Performing Institutions")
    
    top_performers = current_year_data.nlargest(10, 'performance_score')[['institution_name', 'performance_score', 'naac_grade', 'placement_rate']]
    
    fig6 = px.bar(
        top_performers,
        x='performance_score',
        y='institution_name',
        orientation='h',
        title="Top 10 Institutions",
        color='performance_score',
        color_continuous_scale='Viridis',
        hover_data=['naac_grade', 'placement_rate']
    )
    fig6.update_layout(yaxis_title="Institution", xaxis_title="Performance Score")
    st.plotly_chart(fig6, use_container_width=True)

    available_metrics = {
        "Performance Score": "performance_score",
        "Placement Rate (%)": "placement_rate",
        "Research Publications": "research_publications",
        "Student-Faculty Ratio": "student_faculty_ratio",
        "Financial Stability Score": "financial_stability_score",
        "NAAC Grade (Numeric)": "naac_grade",
        "NIRF Ranking": "nirf_ranking",
        "Digital Infrastructure Score": "digital_infrastructure_score",
        "Research Grants Amount": "research_grants_amount",
        "Patents Filed": "patents_filed",
        "Campus Area": "campus_area",
        "Library Volumes": "library_volumes",
    }
    
    # Institution Comparison Tool
    st.subheader("🔍 Compare Institutions")
    
    institutions_list = current_year_data['institution_name'].tolist()
    selected_institutions = st.multiselect(
        "Select institutions to compare:",
        institutions_list,
        default=institutions_list[:3] if len(institutions_list) >= 3 else institutions_list
    )
    
    if selected_institutions:
        comparison_data = current_year_data[current_year_data['institution_name'].isin(selected_institutions)]
        
        selected_metrics = st.multiselect(
            "Select parameters to compare:",
            list(available_metrics.keys()),
            default=["Performance Score", "Placement Rate (%)", "Research Publications"]
        )
        
        fig7 = go.Figure()
        
        for metric_label in selected_metrics:
            metric_col = available_metrics[metric_label]

            fig7.add_trace(go.Bar(
                x=comparison_data['institution_name'],
                y=comparison_data[metric_col],
                name=metric_label
            ))
        
        fig7.update_layout(
            title="Institution Comparison",
            xaxis_title="Institution",
            yaxis_title="Value",
            barmode='group'
        )
        
        st.plotly_chart(fig7, use_container_width=True)
        
        # Detailed comparison table
        comparison_cols = ["institution_name"] + [available_metrics[m] for m in selected_metrics]

        st.dataframe(
            comparison_data[comparison_cols].set_index('institution_name'),
            use_container_width=True    
        )
    
    # Performance Score Calculator
    st.subheader("🧮 Performance Score Calculator")
    
    with st.expander("Calculate Custom Performance Score"):
        col1, col2 = st.columns(2)
        
        with col1:
            naac_grade = st.selectbox(
                "NAAC Grade",
                ["A++", "A+", "A", "B++", "B+", "B", "C"],
                index=2
            )
            naac_score = {"A++":4.0, "A+":3.7, "A":3.4, "B++":3.1, "B+":2.8, "B":2.5, "C":2.0}[naac_grade]
            nirf_rank = st.number_input(
                "NIRF Ranking",
                min_value=1,
                max_value=200,
                value=50,
                help="Leave as 200 if not ranked"
            )
            student_faculty_ratio = st.slider(
                "Student-Faculty Ratio",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )
        
        with col2:
            placement_rate = st.slider(
                "Placement Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )
            research_publications = st.number_input(
                "Research Publications",
                min_value=0,
                value=50,
                step=10
            )
            digital_infrastructure = st.slider(
                "Digital Infrastructure Score",
                min_value=1.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )
        
        if st.button("Calculate Performance Score"):
            from core.database import calculate_performance_score
            
            metrics_dict = {
                # NAAC numeric conversion (A++=4.0 ... C=1.0)
                "naac_grade": naac_score,

                # NIRF Ranking (None if invalid)
                "nirf_ranking": nirf_ranking if nirf_ranking and nirf_ranking <= 200 else None,

                # Faculty ratios
                "student_faculty_ratio": student_faculty_ratio,
                "phd_faculty_ratio": phd_faculty_ratio,

                # Publications, faculty count, pubs/faculty
                "research_publications": research_publications,
                "faculty_count": faculty_count,
                "publications_per_faculty": (
                    research_publications / faculty_count
                    if faculty_count and faculty_count > 0 else 0
                ),

                # Research grants
                "research_grants_amount": research_grants_amount,

                # Patents
                "patents_filed": patents_filed,

                # Infrastructure
                "digital_infrastructure_score": digital_infrastructure_score,
                "library_volumes": library_volumes,
                "campus_area": campus_area,

                # Finance
                "financial_stability_score": financial_stability_score,
                "annual_budget": annual_budget,
                "research_investment": research_investment,

                # Placement normalization
                "placement_rate": (
                    placement_rate / 100 if placement_rate > 1 else placement_rate
                ),

                # Community engagement
                "community_projects": community_projects or 0
            }
            
            score = calculate_performance_score(metrics_dict)
            
            st.success(f"**Calculated Performance Score: {score:.2f}/10**")
            
            # Show recommendation
            if score >= 8.0:
                st.info("**Approval Recommendation:** Full Approval - 5 Years")
                st.success("**Risk Level:** Low Risk")
            elif score >= 7.0:
                st.info("**Approval Recommendation:** Provisional Approval - 3 Years")
                st.info("**Risk Level:** Medium Risk")
            elif score >= 6.0:
                st.warning("**Approval Recommendation:** Conditional Approval - 1 Year")
                st.warning("**Risk Level:** Medium Risk")
            elif score >= 5.0:
                st.warning("**Approval Recommendation:** Approval with Strict Monitoring - 1 Year")
                st.error("**Risk Level:** High Risk")
            else:
                st.error("**Approval Recommendation:** Rejection - Significant Improvements Required")
                st.error("**Risk Level:** Critical Risk")
    
    # Download Options
    st.subheader("📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = current_year_data.to_csv(index=False)
        st.download_button(
            label="Download Current Year Data (CSV)",
            data=csv_data,
            file_name="institutions_2023.csv",
            mime="text/csv"
        )
    
    with col2:
        full_csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download All Historical Data (CSV)",
            data=full_csv_data,
            file_name="institutions_all_years.csv",
            mime="text/csv"
        )










