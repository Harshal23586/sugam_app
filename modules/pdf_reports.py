# modules/pdf_reports.py
import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import os
from io import BytesIO
import zipfile

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
                # Generate the report - FIXED: returns (filepath, pdf_bytes)
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
