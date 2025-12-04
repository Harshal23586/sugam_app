# modules/pdf_reports.py
import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import os
from io import BytesIO
import zipfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def generate_institutional_report(self, institution_id, report_type="comprehensive"):
        """Generate PDF report for an institution with different report types"""
        
        # Get institution data
        institution_data = self.analyzer.historical_data[
            self.analyzer.historical_data['institution_id'] == institution_id
        ]
        
        if institution_data.empty:
            raise ValueError(f"Institution {institution_id} not found")
        
        latest_data = institution_data[institution_data['year'] == institution_data['year'].max()].iloc[0]
        
        # Call appropriate method based on report type
        if report_type == "executive":
            return self._generate_executive_report(latest_data, institution_id)
        elif report_type == "detailed":
            return self._generate_detailed_report(latest_data, institution_id)
        elif report_type == "approval":
            return self._generate_approval_report(latest_data, institution_id)
        else:  # comprehensive (default)
            return self._generate_comprehensive_report(latest_data, institution_id)
    
    def _generate_executive_report(self, institution_data, institution_id):
        """Generate executive summary report"""
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
        story.append(Paragraph("UGC/AICTE EXECUTIVE SUMMARY REPORT", title_style))
        story.append(Spacer(1, 12))
        
        # Institution details
        story.append(Paragraph(f"Institution: {institution_data['institution_name']}", styles['Normal']))
        story.append(Paragraph(f"Institution ID: {institution_id}", styles['Normal']))
        story.append(Paragraph(f"Report Type: Executive Summary", styles['Normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Key metrics at a glance
        story.append(Paragraph("KEY METRICS AT A GLANCE", heading_style))
        
        # Create metrics table
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Performance Score', f"{institution_data['performance_score']:.2f}/10", 
             self._get_status_symbol(institution_data['performance_score'])],
            ['Risk Level', institution_data['risk_level'], 
             self._get_risk_symbol(institution_data['risk_level'])],
            ['Approval Recommendation', institution_data['approval_recommendation'], 
             self._get_approval_symbol(institution_data['approval_recommendation'])],
            ['NAAC Grade', institution_data.get('naac_grade', 'N/A'), 
             self._get_grade_symbol(institution_data.get('naac_grade', 'N/A'))]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Executive summary text
        story.append(Paragraph("EXECUTIVE OVERVIEW", heading_style))
        
        summary_text = f"""
        This executive summary provides a high-level overview of {institution_data['institution_name']}'s 
        performance. The institution has achieved a score of {institution_data['performance_score']:.2f} out of 10 
        and is categorized as {institution_data['risk_level'].lower()} risk. The assessment indicates 
        {institution_data['approval_recommendation'].lower()}.
        
        Key strengths include {self._get_top_strength(institution_data)}. The institution demonstrates 
        solid performance in core metrics but may benefit from improvements in certain areas.
        
        For detailed analysis and comprehensive assessment, please refer to the full institutional report.
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("TOP RECOMMENDATIONS", heading_style))
        recommendations = self._get_recommendations(institution_data, limit=3)
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return self._save_report(pdf_bytes, institution_id, "executive")
    
    def _generate_detailed_report(self, institution_data, institution_id):
        """Generate detailed analytical report"""
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
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#00695c')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#004d40')
        )
        
        # Add header
        story.append(Paragraph("DETAILED ANALYTICAL REPORT", title_style))
        story.append(Spacer(1, 12))
        
        # Institution details
        story.append(Paragraph(f"Institution: {institution_data['institution_name']}", styles['Normal']))
        story.append(Paragraph(f"Detailed Analysis Report", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Detailed metrics section
        story.append(Paragraph("COMPREHENSIVE METRICS ANALYSIS", heading_style))
        
        # Academic metrics
        story.append(Paragraph("Academic Performance Metrics", styles['Heading3']))
        academic_data = [
            ['Metric', 'Value', 'Benchmark', 'Status'],
            ['NAAC Grade', institution_data.get('naac_grade', 'N/A'), 'A+', self._get_grade_status(institution_data.get('naac_grade', 'N/A'))],
            ['NIRF Ranking', institution_data.get('nirf_ranking', 'N/A'), 'Top 100', self._get_ranking_status(institution_data.get('nirf_ranking', 'N/A'))],
            ['Student-Faculty Ratio', f"{institution_data.get('student_faculty_ratio', 0):.1f}:1", '15:1', self._get_ratio_status(institution_data.get('student_faculty_ratio', 0))],
            ['PhD Faculty Ratio', f"{institution_data.get('phd_faculty_ratio', 0):.1f}%", '70%', self._get_percentage_status(institution_data.get('phd_faculty_ratio', 0), 70)],
            ['Research Publications', institution_data.get('research_publications', 0), '50+', self._get_publication_status(institution_data.get('research_publications', 0))]
        ]
        
        academic_table = Table(academic_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        academic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00695c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e0f2f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(academic_table)
        story.append(Spacer(1, 15))
        
        # Placement metrics
        story.append(Paragraph("Student Outcome Metrics", styles['Heading3']))
        placement_data = [
            ['Metric', 'Value', 'Benchmark', 'Status'],
            ['Placement Rate', f"{institution_data.get('placement_rate', 0):.1f}%", '80%', self._get_percentage_status(institution_data.get('placement_rate', 0), 80)],
            ['Average Salary', f"₹{institution_data.get('average_salary', 0):,.0f}", '₹500,000', self._get_salary_status(institution_data.get('average_salary', 0))],
            ['Higher Studies Rate', f"{institution_data.get('higher_studies_rate', 0):.1f}%", '20%', self._get_percentage_status(institution_data.get('higher_studies_rate', 0), 20)],
            ['Entrepreneurship Rate', f"{institution_data.get('entrepreneurship_rate', 0):.1f}%", '5%', self._get_percentage_status(institution_data.get('entrepreneurship_rate', 0), 5)]
        ]
        
        placement_table = Table(placement_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        placement_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00695c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e0f2f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(placement_table)
        story.append(Spacer(1, 20))
        
        # Detailed analysis
        story.append(Paragraph("DETAILED ANALYSIS", heading_style))
        
        analysis_text = f"""
        This detailed analytical report provides an in-depth examination of {institution_data['institution_name']}'s 
        performance across multiple dimensions. Each metric has been evaluated against established benchmarks 
        and industry standards.
        
        The institution's overall performance score of {institution_data['performance_score']:.2f}/10 is derived from 
        weighted analysis of academic quality, student outcomes, research output, infrastructure, and financial 
        stability. The risk assessment of '{institution_data['risk_level']}' reflects the institution's stability 
        and compliance with regulatory requirements.
        
        Key findings from the detailed analysis include strengths in specific areas and identified opportunities 
        for improvement. The following sections provide comprehensive insights and data-driven recommendations.
        """
        
        story.append(Paragraph(analysis_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return self._save_report(pdf_bytes, institution_id, "detailed")
    
    def _generate_approval_report(self, institution_data, institution_id):
        """Generate official approval report"""
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
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#b71c1c'),
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#c62828')
        )
        
        # Official header
        story.append(Paragraph("OFFICIAL APPROVAL REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Official notice
        official_text = f"""
        <b>UGC/AICTE - UNIVERSITY GRANTS COMMISSION</b><br/>
        <b>ALL INDIA COUNCIL FOR TECHNICAL EDUCATION</b><br/><br/>
        
        <b>REFERENCE NO:</b> UGC/AICTE/APR/{datetime.now().year}/{institution_id}<br/>
        <b>DATE:</b> {datetime.now().strftime('%d %B %Y')}<br/><br/>
        
        <b>TO:</b> {institution_data['institution_name']}<br/>
        <b>SUBJECT:</b> Institutional Approval Status<br/><br/>
        """
        
        story.append(Paragraph(official_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Approval decision
        story.append(Paragraph("APPROVAL DECISION", heading_style))
        
        decision_text = f"""
        Based on comprehensive evaluation and assessment conducted by the SUGAM system, 
        the approval status for <b>{institution_data['institution_name']}</b> is determined as follows:
        
        <b>Institution ID:</b> {institution_id}<br/>
        <b>Performance Score:</b> {institution_data['performance_score']:.2f}/10.0<br/>
        <b>Risk Assessment:</b> {institution_data['risk_level']}<br/>
        <b>Official Recommendation:</b> {institution_data['approval_recommendation']}<br/><br/>
        
        <b>OFFICIAL DECISION:</b> {self._get_approval_decision(institution_data['approval_recommendation'])}
        """
        
        story.append(Paragraph(decision_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Compliance checklist
        story.append(Paragraph("COMPLIANCE CHECKLIST", heading_style))
        
        compliance_data = [
            ['Requirement', 'Status', 'Comments'],
            ['Accreditation Status', '✓ Compliant' if institution_data.get('naac_grade') else '⚠️ Pending', institution_data.get('naac_grade', 'N/A')],
            ['Faculty Qualifications', self._get_faculty_status(institution_data), f"PhD Ratio: {institution_data.get('phd_faculty_ratio', 0):.1f}%"],
            ['Infrastructure', self._get_infrastructure_status(institution_data), f"Score: {institution_data.get('digital_infrastructure_score', 0):.1f}/10"],
            ['Financial Stability', self._get_financial_status(institution_data), f"Score: {institution_data.get('financial_stability_score', 0):.1f}/10"],
            ['Student Outcomes', self._get_outcomes_status(institution_data), f"Placement: {institution_data.get('placement_rate', 0):.1f}%"],
            ['Research Output', self._get_research_status(institution_data), f"Publications: {institution_data.get('research_publications', 0)}"]
        ]
        
        compliance_table = Table(compliance_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        compliance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#b71c1c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffebee')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(compliance_table)
        story.append(Spacer(1, 20))
        
        # Official remarks
        story.append(Paragraph("OFFICIAL REMARKS", heading_style))
        
        remarks_text = f"""
        This official approval report constitutes the formal decision based on the comprehensive 
        assessment conducted. The institution is required to adhere to all regulatory requirements 
        and maintain the standards indicated in this report.
        
        The approval status is valid for the period specified in the recommendation. Regular 
        monitoring and compliance reporting will be required as per UGC/AICTE regulations.
        
        For any queries or clarifications regarding this approval report, please contact the 
        designated authority at UGC/AICTE.
        """
        
        story.append(Paragraph(remarks_text, styles['Normal']))
        
        # Official signature section
        story.append(Spacer(1, 30))
        story.append(Paragraph("_________________________", styles['Normal']))
        story.append(Paragraph("<b>Authorized Signatory</b>", styles['Normal']))
        story.append(Paragraph("UGC/AICTE Approval Committee", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return self._save_report(pdf_bytes, institution_id, "approval")
    
    def _generate_comprehensive_report(self, institution_data, institution_id):
        """Generate comprehensive report"""
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
        story.append(Paragraph("UGC/AICTE COMPREHENSIVE ASSESSMENT REPORT", title_style))
        story.append(Spacer(1, 12))
        
        # Institution details
        story.append(Paragraph(f"Institution: {institution_data['institution_name']}", styles['Normal']))
        story.append(Paragraph(f"Institution ID: {institution_id}", styles['Normal']))
        story.append(Paragraph(f"Report Type: Comprehensive Assessment", styles['Normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Combine all sections
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        summary_text = f"""
        This comprehensive report provides a complete assessment of {institution_data['institution_name']}'s 
        performance. The institution has achieved a score of {institution_data['performance_score']:.2f} out of 10 
        and is categorized as {institution_data['risk_level'].lower()} risk. The assessment indicates 
        {institution_data['approval_recommendation'].lower()}.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("DETAILED ANALYSIS", heading_style))
        
        analysis_text = f"""
        The institution demonstrates {self._get_top_strength(institution_data)}. This comprehensive 
        analysis evaluates all key performance indicators against established benchmarks to provide 
        a holistic view of institutional performance.
        """
        story.append(Paragraph(analysis_text, styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("KEY RECOMMENDATIONS", heading_style))
        recommendations = self._get_recommendations(institution_data, limit=5)
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return self._save_report(pdf_bytes, institution_id, "comprehensive")
    
    # Helper methods for status indicators
    def _get_status_symbol(self, score):
        if score >= 8:
            return "✓ Excellent"
        elif score >= 6:
            return "✓ Good"
        elif score >= 5:
            return "⚠️ Fair"
        else:
            return "❌ Needs Improvement"
    
    def _get_risk_symbol(self, risk_level):
        symbols = {
            'Low': '✅ Low Risk',
            'Medium': '⚠️ Medium Risk',
            'High': '⚠️ High Risk',
            'Critical': '❌ Critical Risk'
        }
        return symbols.get(risk_level, '⚠️ Unknown Risk')
    
    def _get_approval_symbol(self, approval):
        if 'Full' in approval:
            return "✅ Approved"
        elif 'Provisional' in approval:
            return "⚠️ Provisional"
        elif 'Conditional' in approval:
            return "⚠️ Conditional"
        elif 'Rejection' in approval:
            return "❌ Rejected"
        else:
            return "⏳ Pending"
    
    def _get_grade_symbol(self, grade):
        if grade in ['A++', 'A+', 'A']:
            return "✓ Excellent"
        elif grade in ['B++', 'B+', 'B']:
            return "✓ Good"
        elif grade in ['C']:
            return "⚠️ Fair"
        else:
            return "❌ Not Accredited"
    
    def _get_top_strength(self, institution_data):
        # Simple logic to determine top strength
        if institution_data.get('placement_rate', 0) > 85:
            return "strong placement outcomes"
        elif institution_data.get('naac_grade', '') in ['A++', 'A+']:
            return "excellent accreditation standing"
        elif institution_data.get('research_publications', 0) > 100:
            return "significant research output"
        else:
            return "consistent performance across metrics"
    
    def _get_recommendations(self, institution_data, limit=5):
        # Generate recommendations based on data
        recommendations = []
        
        if institution_data.get('placement_rate', 0) < 80:
            recommendations.append("Improve placement rates through enhanced industry partnerships")
        
        if institution_data.get('student_faculty_ratio', 0) > 20:
            recommendations.append("Optimize student-faculty ratio to improve teaching quality")
        
        if institution_data.get('research_publications', 0) < 50:
            recommendations.append("Increase research output through faculty development programs")
        
        if institution_data.get('digital_infrastructure_score', 0) < 7:
            recommendations.append("Enhance digital infrastructure and e-learning facilities")
        
        if institution_data.get('financial_stability_score', 0) < 7:
            recommendations.append("Strengthen financial planning and resource management")
        
        return recommendations[:limit]
    
    # Additional helper methods needed for the detailed report
    def _get_grade_status(self, grade):
        if grade in ['A++', 'A+', 'A']:
            return "✓ Excellent"
        elif grade in ['B++', 'B+', 'B']:
            return "✓ Good"
        else:
            return "⚠️ Needs Improvement"
    
    def _get_ranking_status(self, ranking):
        try:
            if ranking == 'N/A':
                return "❌ Not Ranked"
            if isinstance(ranking, str) and ranking.isdigit():
                rank = int(ranking)
                if rank <= 100:
                    return "✓ Excellent"
                elif rank <= 200:
                    return "✓ Good"
                else:
                    return "⚠️ Needs Improvement"
        except:
            pass
        return "⚠️ Data Not Available"
    
    def _get_ratio_status(self, ratio):
        if ratio <= 15:
            return "✓ Excellent"
        elif ratio <= 20:
            return "✓ Good"
        elif ratio <= 25:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_percentage_status(self, percentage, benchmark):
        if percentage >= benchmark:
            return "✓ Above Benchmark"
        elif percentage >= benchmark * 0.8:
            return "✓ Near Benchmark"
        elif percentage >= benchmark * 0.6:
            return "⚠️ Below Benchmark"
        else:
            return "❌ Well Below"
    
    def _get_publication_status(self, publications):
        if publications >= 100:
            return "✓ Excellent"
        elif publications >= 50:
            return "✓ Good"
        elif publications >= 25:
            return "⚠️ Fair"
        else:
            return "❌ Low"
    
    def _get_salary_status(self, salary):
        if salary >= 600000:
            return "✓ Excellent"
        elif salary >= 400000:
            return "✓ Good"
        elif salary >= 300000:
            return "⚠️ Fair"
        else:
            return "❌ Low"
    
    def _get_approval_decision(self, approval_recommendation):
        if 'Full Approval' in approval_recommendation:
            return "GRANTED - Full approval for 5 years"
        elif 'Provisional' in approval_recommendation:
            return "GRANTED - Provisional approval for 3 years"
        elif 'Conditional' in approval_recommendation:
            return "GRANTED - Conditional approval for 1 year"
        elif 'Rejection' in approval_recommendation:
            return "REJECTED - Does not meet minimum requirements"
        else:
            return "PENDING FURTHER REVIEW"
    
    def _get_faculty_status(self, institution_data):
        ratio = institution_data.get('phd_faculty_ratio', 0)
        if ratio >= 70:
            return "✓ Excellent"
        elif ratio >= 50:
            return "✓ Good"
        elif ratio >= 30:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_infrastructure_status(self, institution_data):
        score = institution_data.get('digital_infrastructure_score', 0)
        if score >= 8:
            return "✓ Excellent"
        elif score >= 6:
            return "✓ Good"
        elif score >= 4:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_financial_status(self, institution_data):
        score = institution_data.get('financial_stability_score', 0)
        if score >= 8:
            return "✓ Excellent"
        elif score >= 6:
            return "✓ Good"
        elif score >= 4:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_outcomes_status(self, institution_data):
        rate = institution_data.get('placement_rate', 0)
        if rate >= 85:
            return "✓ Excellent"
        elif rate >= 70:
            return "✓ Good"
        elif rate >= 50:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_research_status(self, institution_data):
        publications = institution_data.get('research_publications', 0)
        if publications >= 100:
            return "✓ Excellent"
        elif publications >= 50:
            return "✓ Good"
        elif publications >= 25:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _save_report(self, pdf_bytes, institution_id, report_type):
        """Save report to file and return filepath"""
        report_dir = "data/reports"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{institution_id}_{report_type}_report_{timestamp}.pdf"
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(pdf_bytes)
        
        return filepath, pdf_bytes


# This is the function that will be imported in main.py
def create_pdf_report_module(analyzer):
    """Main function for PDF report generation module"""
    st.header("📄 PDF Report Generation")
    
    st.info("Generate professional PDF reports for institutional assessments and approvals")
    
    # Check if report_generator exists
    if not hasattr(analyzer, 'report_generator') or analyzer.report_generator is None:
        st.error("❌ PDF Report Generator not initialized")
        st.info("Please make sure the PDFReportGenerator is properly initialized in the analyzer")
        return
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    
    if current_institutions.empty:
        st.warning("No institution data available for 2023")
        return
    
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
                pdf_path, pdf_bytes = analyzer.report_generator.generate_institutional_report(
                    selected_institution_id,
                    selected_type
                )
                
                # Provide download button
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
                    # Email option (future enhancement)
                    if st.button("📧 Email Report"):
                        st.info("Email functionality would be integrated here")
                
                # Report details
                st.info(f"**Report Details:**")
                st.write(f"- **File:** {os.path.basename(pdf_path)}")
                st.write(f"- **Size:** {len(pdf_bytes) / 1024:.1f} KB")
                st.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Type:** {report_type}")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.exception(e)
    
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
                        pdf_path, pdf_bytes = analyzer.report_generator.generate_institutional_report(
                            inst_id,
                            "executive" if batch_report_type == "Executive Summary" else "comprehensive"
                        )
                        generated_reports.append((inst_display, pdf_path, pdf_bytes))
                    except Exception as e:
                        st.warning(f"Failed to generate report for {inst_display}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_institutions))
                
                # Create zip file of all reports
                if generated_reports:
                    zip_buffer = BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for inst_display, pdf_path, pdf_bytes in generated_reports:
                            zip_file.writestr(
                                os.path.basename(pdf_path),
                                pdf_bytes
                            )
                    
                    zip_buffer.seek(0)
                    
                    st.success(f"✅ Generated {len(generated_reports)} reports successfully!")
                    
                    st.download_button(
                        label="📦 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"institutional_reports_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip"
                    )
