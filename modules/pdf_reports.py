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
        """Generate comprehensive report (original implementation)"""
        # This should be your original generate_institutional_report method
        # Rename the existing method to this and update it to work standalone
        
        # [Your original implementation goes here]
        # Make sure to rename your current generate_institutional_report method to this
    
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
