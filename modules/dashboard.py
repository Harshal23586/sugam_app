import streamlit as st
import plotly.express as px
import pandas as pd


# Existing analytical modules (unchanged)
def create_performance_dashboard(analyzer):
    st.header("📊 Institutional Performance Analytics Dashboard")
    
    # Use the actual data from analyzer (should be 20 institutions × 10 years)
    df = analyzer.historical_data
    
    # Show data specification at the top
    st.info(f"📊 **Data Overview**: {df['institution_id'].nunique()} Institutions × {df['year'].nunique()} Years ({df['year'].min()}-{df['year'].max()}) | Total Records: {len(df)}")
    
    # Filter for current year data only for KPI calculations
    current_year_data = df[df['year'] == 2023]
    
    # Ensure we have data
    if len(current_year_data) == 0:
        st.warning("No data available for the current year. Please check data generation.")
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
    
    # Performance Analysis
    st.subheader("📈 Performance Analysis (20 Institutions)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance Distribution for current year
        fig1 = px.histogram(
            current_year_data, 
            x='performance_score',
            title="Distribution of Institutional Performance Scores (2023)",
            nbins=12,
            color_discrete_sequence=['#1f77b4'],
            opacity=0.8
        )
        fig1.update_layout(
            xaxis_title="Performance Score", 
            yaxis_title="Number of Institutions",
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Performance by Institution Type
        fig2 = px.box(
            current_year_data,
            x='institution_type',
            y='performance_score',
            title="Performance Score by Institution Type (2023)",
            color='institution_type'
        )
        fig2.update_layout(
            xaxis_title="Institution Type",
            yaxis_title="Performance Score",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Trend Analysis - Show all 10 years for the 20 institutions
    st.subheader("📅 Historical Performance Trends (2014-2023)")
    
    # Calculate average performance by year and type
    trend_data = df.groupby(['year', 'institution_type'])['performance_score'].mean().reset_index()
    
    fig3 = px.line(
        trend_data,
        x='year',
        y='performance_score',
        color='institution_type',
        title="Average Performance Score Trend (2014-2023) - 20 Institutions",
        markers=True
    )
    fig3.update_layout(
        xaxis_title="Year", 
        yaxis_title="Average Performance Score",
        legend_title="Institution Type"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Risk Analysis
    st.subheader("⚠️ Institutional Risk Analysis (2023)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_distribution = current_year_data['risk_level'].value_counts()
        fig4 = px.pie(
            values=risk_distribution.values,
            names=risk_distribution.index,
            title="Institutional Risk Level Distribution",
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
        # Placement vs Research Analysis
        fig5 = px.scatter(
            current_year_data,
            x='research_publications',
            y='placement_rate',
            color='risk_level',
            size='performance_score',
            hover_data=['institution_name'],
            title="Research Output vs Placement Rate (2023)",
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c',
                'Critical Risk': '#c0392b'
            }
        )
        fig5.update_layout(
            xaxis_title="Research Publications",
            yaxis_title="Placement Rate (%)"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Additional Visualizations focused on 20 institutions
    st.subheader("🎯 Performance Insights - 20 Institutions Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 performing institutions
        top_performers = current_year_data.nlargest(10, 'performance_score')[['institution_name', 'performance_score', 'naac_grade']]
        fig6 = px.bar(
            top_performers,
            x='performance_score',
            y='institution_name',
            orientation='h',
            title="Top 10 Performing Institutions (2023)",
            color='performance_score',
            color_continuous_scale='Viridis'
        )
        fig6.update_layout(
            yaxis_title="Institution",
            xaxis_title="Performance Score"
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # State-wise Performance
        state_performance = current_year_data.groupby('state')['performance_score'].mean().sort_values(ascending=False)
        fig7 = px.bar(
            x=state_performance.index,
            y=state_performance.values,
            title="States by Average Performance Score (2023)",
            color=state_performance.values,
            color_continuous_scale='Viridis'
        )
        fig7.update_layout(
            xaxis_title="State",
            yaxis_title="Average Performance Score",
            showlegend=False
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    # Comprehensive Data Table
    st.subheader("📋 Institutional Performance Data (2023)")
    
    # Show key metrics for all 20 institutions
    display_columns = [
        'institution_id', 'institution_name', 'institution_type', 'state',
        'performance_score', 'naac_grade', 'placement_rate', 'risk_level',
        'approval_recommendation'
    ]
    
    st.dataframe(
        current_year_data[display_columns].sort_values('performance_score', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Institution Comparison Tool
    st.subheader("🔍 Compare Institutions")
    
    # Select institutions to compare
    institutions_list = current_year_data['institution_name'].tolist()
    selected_institutions = st.multiselect(
        "Select institutions to compare:",
        institutions_list,
        default=institutions_list[:3] if len(institutions_list) >= 3 else institutions_list
    )
    
    if selected_institutions:
        comparison_data = current_year_data[current_year_data['institution_name'].isin(selected_institutions)]
        
        # Create comparison chart
        fig8 = px.bar(
            comparison_data,
            x='institution_name',
            y=['performance_score', 'placement_rate', 'research_publications'],
            title="Institution Comparison",
            barmode='group'
        )
        fig8.update_layout(
            xaxis_title="Institution",
            yaxis_title="Score/Percentage/Count"
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Show detailed comparison table
        comparison_cols = [
            'institution_name', 'performance_score', 'naac_grade', 'nirf_ranking',
            'placement_rate', 'research_publications', 'student_faculty_ratio',
            'financial_stability_score', 'risk_level'
        ]
        st.dataframe(
            comparison_data[comparison_cols].set_index('institution_name'),
            use_container_width=True
        )
