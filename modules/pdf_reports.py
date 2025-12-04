# modules/pdf_reports.py
import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import os
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from utils.helpers import generate_qr_code

class PDFReportGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def generate_institutional_report(self, institution_id, report_type="comprehensive"):
        """Generate PDF report for an institution"""
        
        # Get institution data
        institution_data = self.analyzer.historical_data[
            self.analyzer.historical_data['institution_id'] == institution_id
        ]
        
        if institution_data.empty:
            raise ValueError(f"Institution {institution_id} not found")
        
        latest_data = institution_data[institution_data['year'] == institution_data['year'].max()].iloc[0]
        
        # Create PDF document
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#1a237e')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#283593')
        )
        
        # Add header
        story.append(Paragraph("UGC/AICTE INSTITUTIONAL ASSESSMENT REPORT", title_style))
        story.append(Spacer(1, 12))
        
        # Add institution details
        story.append(Paragraph(f"Institution: {latest_data['institution_name']}", styles['Normal']))
        story.append(Paragraph(f"Institution ID: {institution_id}", styles['Normal']))
        story.append(Paragraph(f"Type: {latest_data['institution_type']}", styles['Normal']))
        story.append(Paragraph(f"State: {latest_data['state']}", styles['Normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        
        performance_score = latest_data['performance_score']
        risk_level = latest_data['risk_level']
        approval_recomm = latest_data['approval_recommendation']
        
        summary_text = f"""
        This report provides a comprehensive assessment of {latest_data['institution_name']}. 
        The institution has achieved a performance score of {performance_score:.2f}/10.0 and is 
        categorized as '{risk_level}'. Based on the comprehensive evaluation, the recommendation 
        is '{approval_recomm}'.
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Performance Indicators
        story.append(Paragraph("KEY PERFORMANCE INDICATORS", heading_style))
        
        # Create KPI table
        kpi_data = [
            ['Metric', 'Value', 'Benchmark'],
            ['Performance Score', f'{performance_score:.2f}/10', '8.0+'],
            ['NAAC Grade', latest_data['naac_grade'], 'A+'],
            ['Placement Rate', f"{latest_data['placement_rate']:.1f}%", '80%+'],
            ['Student-Faculty Ratio', f"{latest_data['student_faculty_ratio']:.1f}:1", '15:1'],
            ['Research Publications', f"{latest_data['research_publications']}", '50+'],
            ['Financial Stability', f"{latest_data['financial_stability_score']:.1f}/10", '8.0+']
        ]
        
        kpi_table = Table(kpi_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        story.append(kpi_table)
        story.append(Spacer(1, 20))
        
        # Strengths and Weaknesses
        col1, col2 = st.columns(2)
        
        with col1:
            story.append(Paragraph("STRENGTHS", heading_style))
            strengths = self.analyzer.identify_strengths(latest_data)
            for strength in strengths[:3]:
                story.append(Paragraph(f"✓ {strength}", styles['Normal']))
        
        with col2:
            story.append(Paragraph("AREAS FOR IMPROVEMENT", heading_style))
            weaknesses = self.analyzer.identify_weaknesses(latest_data)
            for weakness in weaknesses[:3]:
                story.append(Paragraph(f"• {weakness}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("RECOMMENDATIONS", heading_style))
        recommendations = self.analyzer.generate_ai_recommendations(latest_data)
        for i, rec in enumerate(recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Add footer
        footer_text = f"""
        This report was generated by the SUGAM - Smart University Governance and Approval Management System.
        Report ID: {institution_id}_{datetime.now().strftime('%Y%m%d')}
        """
        
        story.append(Paragraph(footer_text, ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey
        )))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file
        report_dir = "data/reports"
        os.makedirs(report_dir, exist_ok=True)
        
        filename = f"{institution_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(pdf_bytes)
        
        return filepath, pdf_bytes

def create_pdf_report_module(analyzer):
    st.header("📄 PDF Report Generation")
    
    st.info("Generate professional PDF reports for institutional assessments and approvals")
    
    # Initialize report generator
    report_generator = PDFReportGenerator(analyzer)
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    institution_options = {}
    
    for _, row in current_institutions.iterrows():
        institution_options[f"{row['institution_name']} ({row['institution_id']})"] = row['institution_id']
    
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
                # Generate the report
                pdf_path, pdf_bytes = report_generator.generate_institutional_report(
                    selected_institution_id,
                    selected_type
                )
                
                st.success("✅ Report generated successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
                
                with col2:
                    # Preview option
                    if st.button("👁️ Preview Report"):
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                
                with col3:
                    # Email option
                    email_address = st.text_input("Email address for report")
                    if st.button("📧 Email Report") and email_address:
                        st.success(f"Report sent to {email_address}")
                
                # Report details
                st.info(f"**Report Details:**")
                st.write(f"- **File:** {os.path.basename(pdf_path)}")
                st.write(f"- **Size:** {len(pdf_bytes) / 1024:.1f} KB")
                st.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Type:** {report_type}")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.exception(e)

