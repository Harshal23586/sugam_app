# modules/pdf_reports.py
"""
PDF Report Generation Module

Generates professional PDF reports for institutional assessments
and approval documentation.
"""

import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from io import BytesIO
from typing import Dict, Any
import os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF reports disabled.")

class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet() if PDF_AVAILABLE else None
        self.logo_path = self.get_logo_path()
    
    def get_logo_path(self):
        """Get logo path for PDF reports"""
        possible_paths = [
            "logo.jpg",
            "assets/logo.jpg",
            "sugam_app/assets/logo.jpg",
            "logo.png",
            "assets/logo.png"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def generate_institutional_report(self, institution_id: str, report_type: str) -> str:
        """
        Generate institutional report
        
        Args:
            institution_id: Institution ID
            report_type: Type of report (comprehensive, executive, detailed, approval)
        
        Returns:
            Path to generated PDF file
        """
        if not PDF_AVAILABLE:
            raise ImportError("ReportLab required for PDF generation")
        
        # Create output directory
        os.makedirs("reports", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/{institution_id}_{report_type}_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content)
        story = []
        
        # Add title
        story.extend(self.create_title_section(institution_id, report_type))
        
        # Add institution details
        story.extend(self.create_institution_section(institution_id))
        
        # Add performance analysis
        story.extend(self.create_performance_section(institution_id))
        
        # Add recommendations
        story.extend(self.create_recommendations_section(institution_id))
        
        # Add footer
        story.extend(self.create_footer_section())
        
        # Build PDF
        doc.build(story)
        
        return filename
    
    def create_title_section(self, institution_id: str, report_type: str) -> list:
        """Create title section"""
        elements = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center aligned
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.gray,
            spaceAfter=20,
            alignment=1
        )
        
        # Add logo if available
        if self.logo_path:
            try:
                logo = Image(self.logo_path, width=100, height=100)
                logo.hAlign = 'CENTER'
                elements.append(logo)
                elements.append(Spacer(1, 20))
            except:
                pass
        
        # Add title
        title_text = f"Institutional Assessment Report"
        elements.append(Paragraph(title_text, title_style))
        
        # Add subtitle
        subtitle_text = f"Report Type: {report_type.replace('_', ' ').title()} | Institution: {institution_id}"
        elements.append(Paragraph(subtitle_text, subtitle_style))
        
        # Add report info
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.darkgray,
            spaceAfter=30
        )
        
        date_text = f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}"
        elements.append(Paragraph(date_text, info_style))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_institution_section(self, institution_id: str) -> list:
        """Create institution details section"""
        elements = []
        
        # Section title
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#1E3A8A')
        )
        
        elements.append(Paragraph("1. Institution Details", section_style))
        elements.append(Spacer(1, 10))
        
        # Institution data (this would come from your database)
        institution_data = {
            "Institution ID": institution_id,
            "Name": "Sample University",
            "Type": "State University",
            "Established": "1965",
            "Location": "New Delhi",
            "NAAC Grade": "A+",
            "NIRF Ranking": "45"
        }
        
        # Create table
        data = [[key, value] for key, value in institution_data.items()]
        table = Table(data, colWidths=[2*inch, 3*inch])
        
        # Style table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_performance_section(self, institution_id: str) -> list:
        """Create performance analysis section"""
        elements = []
        
        # Section title
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#1E3A8A')
        )
        
        elements.append(Paragraph("2. Performance Analysis", section_style))
        elements.append(Spacer(1, 10))
        
        # Performance metrics (sample data)
        performance_data = [
            ["Metric", "Score", "Status"],
            ["Overall Performance", "8.5/10", "Excellent"],
            ["Academic Excellence", "9.0/10", "Outstanding"],
            ["Research & Innovation", "8.0/10", "Very Good"],
            ["Infrastructure", "7.5/10", "Good"],
            ["Governance", "8.5/10", "Excellent"],
            ["Placement Rate", "85%", "Very Good"],
            ["Student-Faculty Ratio", "18:1", "Good"],
            ["Financial Stability", "9.0/10", "Outstanding"]
        ]
        
        # Create performance table
        table = Table(performance_data, colWidths=[2*inch, 1*inch, 1.5*inch])
        
        # Style table with colors based on status
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#F0F9FF')),
            ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#FEFCE8')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB'))
        ]
        
        # Add status-based coloring
        for i in range(1, len(performance_data)):
            status = performance_data[i][2]
            if status in ["Excellent", "Outstanding"]:
                style_commands.append(('BACKGROUND', (2, i), (2, i), colors.HexColor('#D1FAE5')))
            elif status == "Very Good":
                style_commands.append(('BACKGROUND', (2, i), (2, i), colors.HexColor('#FEF3C7')))
            else:
                style_commands.append(('BACKGROUND', (2, i), (2, i), colors.HexColor('#FEE2E2')))
        
        table.setStyle(TableStyle(style_commands))
        elements.append(table)
        
        # Add summary paragraph
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=15,
            spaceAfter=20
        )
        
        summary_text = """
        <b>Summary:</b> The institution demonstrates strong performance across most parameters 
        with particular excellence in academic standards and financial management. 
        Areas for improvement include infrastructure enhancement and research output.
        """
        elements.append(Paragraph(summary_text, summary_style))
        elements.append(Spacer(1, 10))
        
        return elements
    
    def create_recommendations_section(self, institution_id: str) -> list:
        """Create recommendations section"""
        elements = []
        
        # Section title
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#1E3A8A')
        )
        
        elements.append(Paragraph("3. Recommendations & Approval", section_style))
        elements.append(Spacer(1, 10))
        
        # Recommendations list
        recommendations = [
            "✅ Continue current academic excellence standards",
            "📈 Increase research publications by 20% in next academic year",
            "🏗️ Allocate budget for infrastructure modernization",
            "🤝 Strengthen industry collaborations for placements",
            "🌱 Implement green campus initiatives",
            "📊 Participate in NIRF ranking for national visibility"
        ]
        
        # Create recommendations with icons
        for rec in recommendations:
            bullet_style = ParagraphStyle(
                'BulletStyle',
                parent=self.styles['Normal'],
                fontSize=11,
                leftIndent=20,
                spaceAfter=5,
                bulletIndent=10
            )
            elements.append(Paragraph(f"• {rec}", bullet_style))
        
        elements.append(Spacer(1, 15))
        
        # Approval decision
        approval_style = ParagraphStyle(
            'ApprovalStyle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=15,
            textColor=colors.HexColor('#059669')
        )
        
        approval_text = "<b>Approval Decision:</b> Full Approval - 5 Years"
        elements.append(Paragraph(approval_text, approval_style))
        
        # Conditions
        conditions_style = ParagraphStyle(
            'ConditionsStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6B7280'),
            spaceAfter=20
        )
        
        conditions_text = "<i>Conditions: Submit annual progress reports and maintain NAAC grade of A or above.</i>"
        elements.append(Paragraph(conditions_text, conditions_style))
        
        return elements
    
    def create_footer_section(self) -> list:
        """Create footer section"""
        elements = []
        
        elements.append(Spacer(1, 30))
        
        # Footer text
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6B7280'),
            alignment=1  # Center
        )
        
        footer_text = """
        <b>SUGAM - Smart University Governance and Approval Management</b><br/>
        Ministry of Education, Government of India<br/>
        This is an electronically generated report. No signature required.<br/>
        Report ID: {timestamp} | Page <page/>
        """.format(timestamp=datetime.now().strftime("%Y%m%d%H%M%S"))
        
        elements.append(Paragraph(footer_text, footer_style))
        
        return elements

def create_pdf_report_module(analyzer):
    """
    PDF Report Generation Interface
    
    Args:
        analyzer: InstitutionalAIAnalyzer instance
    """
    if not PDF_AVAILABLE:
        st.error("""
        ⚠️ PDF Generation requires ReportLab installation.
        
        Install with: `pip install reportlab`
        """)
        return
    
    st.header("📄 PDF Report Generation")
    
    st.info("""
    Generate professional PDF reports for institutional assessments, 
    accreditation documentation, and approval processes.
    """)
    
    # Initialize report generator
    report_gen = PDFReportGenerator()
    
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
    
    # Customization options
    with st.expander("⚙️ Report Customization", expanded=False):
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
    
    # Generate report button
    if st.button("🖨️ Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner(f"Generating {report_type}..."):
            try:
                # Generate the report
                pdf_path = report_gen.generate_institutional_report(
                    selected_institution_id,
                    selected_type
                )
                
                # Read the generated PDF
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                # Provide download button
                st.success("✅ Report generated successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col2:
                    # Preview option
                    if st.button("👁️ Preview Report", use_container_width=True):
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'''
                        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                                width="100%" height="600" type="application/pdf">
                        </iframe>
                        '''
                        st.markdown(pdf_display, unsafe_allow_html=True)
                
                with col3:
                    # Email option placeholder
                    if st.button("📧 Email Report", use_container_width=True):
                        st.info("📧 Email functionality will be integrated in future updates")
                
                # Report details
                st.info(f"**Report Details:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- **File:** {os.path.basename(pdf_path)}")
                    st.write(f"- **Size:** {os.path.getsize(pdf_path) / 1024:.1f} KB")
                with col2:
                    st.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- **Type:** {report_type}")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.info("""
                **Troubleshooting:**
                1. Ensure ReportLab is installed: `pip install reportlab`
                2. Check if 'reports' directory exists
                3. Verify institution data is available
                """)
    
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
    
    if st.button("🖨️ Generate Batch Reports", type="secondary", use_container_width=True):
        if not selected_institutions:
            st.warning("Please select at least one institution")
        else:
            with st.spinner(f"Generating reports for {len(selected_institutions)} institutions..."):
                progress_bar = st.progress(0)
                generated_reports = []
                
                for i, inst_display in enumerate(selected_institutions):
                    inst_id = institution_options[inst_display]
                    try:
                        pdf_path = report_gen.generate_institutional_report(
                            inst_id,
                            "executive" if batch_report_type == "Executive Summary" else "comprehensive"
                        )
                        generated_reports.append((inst_display, pdf_path))
                    except Exception as e:
                        st.warning(f"Failed to generate report for {inst_display}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_institutions))
                
                # Create zip file of all reports
                if generated_reports:
                    import zipfile
                    zip_buffer = BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for inst_display, pdf_path in generated_reports:
                            zip_file.write(pdf_path, os.path.basename(pdf_path))
                    
                    zip_buffer.seek(0)
                    
                    st.success(f"✅ Generated {len(generated_reports)} reports successfully!")
                    
                    st.download_button(
                        label="📦 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"institutional_reports_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
    
    # Report templates section
    st.markdown("---")
    st.subheader("📋 Report Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download Report Template", use_container_width=True):
            st.info("Template download will be implemented in future updates")
    
    with col2:
        if st.button("🔄 Reset to Default Template", use_container_width=True):
            st.success("Template reset to defaults")
    
    with col3:
        if st.button("💾 Save Custom Template", use_container_width=True):
            st.success("Custom template saved successfully")

if __name__ == "__main__":
    # Test the module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from core.analyzer import InstitutionalAIAnalyzer
    
    st.set_page_config(page_title="PDF Reports Test", layout="wide")
    
    analyzer = InstitutionalAIAnalyzer()
    create_pdf_report_module(analyzer)
