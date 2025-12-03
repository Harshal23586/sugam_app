import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import zipfile

def create_pdf_report_module(analyzer):
    st.header("📄 PDF Report Generation")
    
    st.info("Generate professional PDF reports for institutional assessments and approvals")
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    institution_options = {}
    
    for _, row in current_institutions.iterrows():
        institution_options[f"{row['institution_name']} ({row['institution_id']})"] = row['institution_id']
    
    if not institution_options:
        st.warning("No institutions available for report generation")
        return
    
    selected_institution_display = st.selectbox(
        "Select Institution",
        list(institution_options.keys()),
        key="pdf_institution"
    )
    
    selected_institution_id = institution_options[selected_institution_display]
    
    # Report type selection
    report_type = st.radio(
        "Select Report Type",
        ["📋 Comprehensive Report", 
         "🎯 Executive Summary", 
         "📊 Detailed Analytical Report",
         "🏛️ Official Approval Report"],
        horizontal=True
    )
    
    # Map display names to internal types
    report_type_map = {
        "📋 Comprehensive Report": "comprehensive",
        "🎯 Executive Summary": "executive",
        "📊 Detailed Analytical Report": "detailed",
        "🏛️ Official Approval Report": "approval"
    }
    
    selected_type = report_type_map[report_type]
    
    # Show preview of selected institution
    with st.expander("👁️ Institution Preview"):
        institution_data = current_institutions[
            current_institutions['institution_id'] == selected_institution_id
        ].iloc[0]
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.metric("Performance Score", f"{institution_data['performance_score']:.2f}")
            st.metric("Risk Level", institution_data['risk_level'])
    
        with col2:
            st.metric("NAAC Grade", institution_data.get('naac_grade', 'N/A'))
            st.metric("Placement Rate", f"{institution_data.get('placement_rate', 0):.1f}%")
    
        with col3:
            st.metric("Approval Status", institution_data['approval_recommendation'])
            st.metric("Type", institution_data['institution_type'])
    
    # Customization options
    with st.expander("⚙️ Report Customization"):
        col1, col2 = st.columns(2)
    
        with col1:
            include_charts = st.checkbox("Include Charts & Graphs", value=True)
            include_benchmarks = st.checkbox("Include Benchmark Comparisons", value=True)
            confidential = st.checkbox("Confidential Report", value=False)
    
        with col2:
            watermark = st.checkbox("Add UGC/AICTE Watermark", value=True)
            executive_summary = st.checkbox("Add Executive Summary", value=True)
            detailed_appendix = st.checkbox("Include Detailed Appendix", value=False)
    
        # Additional notes
        report_notes = st.text_area(
            "Additional Notes/Comments for Report",
            placeholder="Add any specific comments or observations to include in the report...",
            height=100
        )
    
    # Generate report
    if st.button("🖨️ Generate PDF Report", type="primary"):
        with st.spinner(f"Generating {report_type}..."):
            try:
                # Generate report content
                report_content = generate_report_content(
                    institution_data, 
                    selected_type, 
                    analyzer,
                    include_charts,
                    include_benchmarks
                )
                
                # Create a preview of the report
                st.success("✅ Report generated successfully!")
                
                # Show report preview
                st.subheader("📄 Report Preview")
                
                # Executive Summary
                st.markdown("### 🎯 Executive Summary")
                st.write(f"**Institution:** {institution_data['institution_name']}")
                st.write(f"**Performance Score:** {institution_data['performance_score']:.2f}/10")
                st.write(f"**Risk Level:** {institution_data['risk_level']}")
                st.write(f"**Approval Recommendation:** {institution_data['approval_recommendation']}")
                
                # Performance Metrics
                st.markdown("### 📊 Performance Metrics")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("NAAC Grade", institution_data.get('naac_grade', 'N/A'))
                    st.metric("Placement Rate", f"{institution_data.get('placement_rate', 0):.1f}%")
                
                with metrics_col2:
                    st.metric("Research Publications", institution_data.get('research_publications', 0))
                    st.metric("Student-Faculty Ratio", institution_data.get('student_faculty_ratio', 0))
                
                with metrics_col3:
                    st.metric("Financial Stability", f"{institution_data.get('financial_stability_score', 0):.1f}/10")
                    st.metric("Digital Infrastructure", f"{institution_data.get('digital_infrastructure_score', 0):.1f}/10")
                
                # Performance Chart
                if include_charts:
                    st.markdown("### 📈 Performance Analysis")
                    
                    # Create a simple performance radar chart
                    categories = ['Academic', 'Research', 'Infrastructure', 'Governance', 'Placement']
                    values = [
                        institution_data.get('performance_score', 5) * 0.3,
                        institution_data.get('research_publications', 0) / 10,
                        institution_data.get('digital_infrastructure_score', 5),
                        institution_data.get('financial_stability_score', 5),
                        institution_data.get('placement_rate', 50) / 10
                    ]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Performance'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )),
                        showlegend=True,
                        title="Performance Across Categories"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Benchmark Comparison
                if include_benchmarks:
                    st.markdown("### 🏛️ Peer Comparison")
                    
                    peer_data = analyzer.historical_data[
                        (analyzer.historical_data['institution_type'] == institution_data['institution_type']) &
                        (analyzer.historical_data['year'] == 2023) &
                        (analyzer.historical_data['institution_id'] != selected_institution_id)
                    ]
                    
                    if not peer_data.empty:
                        peer_avg = peer_data['performance_score'].mean()
                        
                        comparison_df = pd.DataFrame({
                            'Metric': ['Performance Score', 'Placement Rate', 'Research Publications'],
                            'Institution': [
                                institution_data['performance_score'],
                                institution_data.get('placement_rate', 0),
                                institution_data.get('research_publications', 0)
                            ],
                            'Peer Average': [
                                peer_avg,
                                peer_data['placement_rate'].mean(),
                                peer_data['research_publications'].mean()
                            ]
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                
                # AI Recommendations
                st.markdown("### 🤖 AI Recommendations")
                
                recommendations = generate_ai_recommendations(institution_data)
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Download options
                st.markdown("---")
                st.subheader("📥 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a simple text report for download
                    report_text = generate_text_report(institution_data, recommendations)
                    
                    st.download_button(
                        label="📥 Download Text Report",
                        data=report_text,
                        file_name=f"report_{selected_institution_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Placeholder for PDF download
                    st.info("**PDF Generation:** Full PDF report generation would require additional libraries like ReportLab or WeasyPrint")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    # Batch report generation
    st.markdown("---")
    st.subheader("🔄 Batch Report Generation")
    
    st.info("Generate reports for multiple institutions at once")
    
    selected_institutions = st.multiselect(
        "Select Institutions for Batch Processing",
        list(institution_options.keys()),
        default=[]
    )
    
    batch_report_type = st.selectbox(
        "Report Type for Batch",
        ["Executive Summary", "Comprehensive Report"],
        key="batch_type"
    )
    
    if st.button("🖨️ Generate Batch Reports", type="secondary"):
        if not selected_institutions:
            st.warning("Please select at least one institution")
        else:
            with st.spinner(f"Generating reports for {len(selected_institutions)} institutions..."):
                progress_bar = st.progress(0)
                generated_reports = []
            
                for i, inst_display in enumerate(selected_institutions):
                    inst_id = institution_options[inst_display]
                    try:
                        # Generate report content
                        inst_data = current_institutions[
                            current_institutions['institution_id'] == inst_id
                        ].iloc[0]
                        
                        report_text = generate_text_report(
                            inst_data, 
                            generate_ai_recommendations(inst_data)
                        )
                        
                        generated_reports.append((inst_display, report_text))
                    except Exception as e:
                        st.warning(f"Failed to generate report for {inst_display}: {str(e)}")
                
                    progress_bar.progress((i + 1) / len(selected_institutions))
            
                # Create zip file of all reports
                if generated_reports:
                    zip_buffer = BytesIO()
                
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for inst_display, report_text in generated_reports:
                            filename = f"report_{inst_display.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
                            zip_file.writestr(filename, report_text)
                
                    zip_buffer.seek(0)
                
                    st.success(f"✅ Generated {len(generated_reports)} reports successfully!")
                
                    st.download_button(
                        label="📦 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"institutional_reports_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip"
                    )

def generate_report_content(institution_data, report_type, analyzer, include_charts, include_benchmarks):
    """Generate report content"""
    # This function would generate the actual report content
    # For now, return a placeholder
    return {
        "executive_summary": f"Report for {institution_data['institution_name']}",
        "performance_data": institution_data.to_dict(),
        "recommendations": generate_ai_recommendations(institution_data)
    }

def generate_text_report(institution_data, recommendations):
    """Generate a text version of the report"""
    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append(f"INSTITUTIONAL PERFORMANCE REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"")
    report_lines.append(f"Institution: {institution_data['institution_name']}")
    report_lines.append(f"Institution ID: {institution_data['institution_id']}")
    report_lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"")
    report_lines.append("-" * 60)
    report_lines.append(f"EXECUTIVE SUMMARY")
    report_lines.append("-" * 60)
    report_lines.append(f"")
    report_lines.append(f"Performance Score: {institution_data['performance_score']:.2f}/10")
    report_lines.append(f"Risk Level: {institution_data['risk_level']}")
    report_lines.append(f"Approval Recommendation: {institution_data['approval_recommendation']}")
    report_lines.append(f"")
    report_lines.append("-" * 60)
    report_lines.append(f"KEY PERFORMANCE INDICATORS")
    report_lines.append("-" * 60)
    report_lines.append(f"")
    report_lines.append(f"NAAC Grade: {institution_data.get('naac_grade', 'N/A')}")
    report_lines.append(f"Placement Rate: {institution_data.get('placement_rate', 0):.1f}%")
    report_lines.append(f"Research Publications: {institution_data.get('research_publications', 0)}")
    report_lines.append(f"Student-Faculty Ratio: {institution_data.get('student_faculty_ratio', 0):.1f}")
    report_lines.append(f"Financial Stability: {institution_data.get('financial_stability_score', 0):.1f}/10")
    report_lines.append(f"")
    report_lines.append("-" * 60)
    report_lines.append(f"AI RECOMMENDATIONS")
    report_lines.append("-" * 60)
    report_lines.append(f"")
    
    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"{i}. {rec}")
    
    report_lines.append(f"")
    report_lines.append("=" * 60)
    report_lines.append(f"End of Report")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)

def generate_ai_recommendations(institution_data):
    """Generate AI recommendations for the institution"""
    recommendations = []
    
    # Student-Faculty Ratio
    sf_ratio = institution_data.get('student_faculty_ratio', 0)
    if sf_ratio > 25:
        recommendations.append("Recruit additional faculty members to improve student-faculty ratio")
    
    # Placement Rate
    placement_rate = institution_data.get('placement_rate', 0)
    if placement_rate < 70:
        recommendations.append("Strengthen industry partnerships and career development programs")
    
    # Research Publications
    research_pubs = institution_data.get('research_publications', 0)
    if research_pubs < 50:
        recommendations.append("Establish research promotion policy and faculty development programs")
    
    # Digital Infrastructure
    digital_score = institution_data.get('digital_infrastructure_score', 0)
    if digital_score < 7:
        recommendations.append("Invest in digital infrastructure and e-learning platforms")
    
    # NAAC Grade
    naac_grade = institution_data.get('naac_grade', '')
    if naac_grade in ['B', 'C']:
        recommendations.append("Work on improving NAAC accreditation through quality enhancement")
    
    # Default recommendations if none apply
    if not recommendations:
        recommendations = [
            "Continue current improvement initiatives",
            "Focus on maintaining quality standards",
            "Explore international collaborations"
        ]
    
    return recommendations[:5]  # Return top 5 recommendations
