# utils/notifications.py
"""
Email Notification System

Handles email notifications for institutional users
"""

import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import json
from datetime import datetime

class EmailNotifier:
    """Handle email notifications"""
    
    def __init__(self):
        self.config = self.load_email_config()
    
    def load_email_config(self) -> Dict:
        """Load email configuration"""
        default_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "noreply@sugam.gov.in",
            "sender_name": "SUGAM System",
            "use_tls": True,
            "test_mode": True  # Set to False in production
        }
        
        # Try to load from config file
        try:
            with open("config/email_config.json", "r") as f:
                return {**default_config, **json.load(f)}
        except:
            return default_config
    
    def send_notification(self, to_email: str, subject: str, 
                         message_type: str, data: Dict) -> bool:
        """
        Send email notification
        
        Args:
            to_email: Recipient email
            subject: Email subject
            message_type: Type of notification
            data: Additional data for template
        
        Returns:
            Success status
        """
        if self.config["test_mode"]:
            # In test mode, show notification in UI instead of sending email
            st.info(f"""
            📧 **Email Notification (Test Mode)**
            
            **To:** {to_email}
            **Subject:** {subject}
            **Type:** {message_type}
            
            Email would be sent in production mode.
            """)
            return True
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.config['sender_name']} <{self.config['sender_email']}>"
            msg['To'] = to_email
            
            # Get message content based on type
            text_content, html_content = self.get_message_content(message_type, data)
            
            # Attach both text and HTML versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config['use_tls']:
                    server.starttls()
                # In production, you would login here
                # server.login(username, password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def get_message_content(self, message_type: str, data: Dict) -> tuple:
        """Get message content based on type"""
        
        templates = {
            "submission_received": {
                "text": """
                Dear {contact_person},
                
                Your {submission_type} has been received for {institution_name}.
                
                Submission ID: {submission_id}
                Date: {submission_date}
                Status: Under Review
                
                You can track your submission at: {portal_url}
                
                Regards,
                SUGAM System
                """,
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color: #1E3A8A;">Submission Received</h2>
                    <p>Dear {contact_person},</p>
                    <p>Your <strong>{submission_type}</strong> has been received for <strong>{institution_name}</strong>.</p>
                    
                    <div style="background: #F3F4F6; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <p><strong>Submission ID:</strong> {submission_id}</p>
                        <p><strong>Date:</strong> {submission_date}</p>
                        <p><strong>Status:</strong> <span style="color: #F59E0B;">Under Review</span></p>
                    </div>
                    
                    <p>You can track your submission at: <a href="{portal_url}">{portal_url}</a></p>
                    
                    <p>Regards,<br/>
                    <strong>SUGAM System</strong></p>
                </body>
                </html>
                """
            },
            "document_uploaded": {
                "text": """
                Dear {contact_person},
                
                Documents have been uploaded for {institution_name}.
                
                Upload Date: {upload_date}
                Number of Documents: {document_count}
                Status: Processing
                
                You can view document status at: {portal_url}
                
                Regards,
                SUGAM System
                """,
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color: #1E3A8A;">Documents Uploaded</h2>
                    <p>Dear {contact_person},</p>
                    <p>Documents have been uploaded for <strong>{institution_name}</strong>.</p>
                    
                    <div style="background: #F3F4F6; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <p><strong>Upload Date:</strong> {upload_date}</p>
                        <p><strong>Number of Documents:</strong> {document_count}</p>
                        <p><strong>Status:</strong> <span style="color: #10B981;">Processing</span></p>
                    </div>
                    
                    <p>You can view document status at: <a href="{portal_url}">{portal_url}</a></p>
                    
                    <p>Regards,<br/>
                    <strong>SUGAM System</strong></p>
                </body>
                </html>
                """
            }
        }
        
        template = templates.get(message_type, templates["submission_received"])
        
        # Format templates with data
        text_content = template["text"].format(**data)
        html_content = template["html"].format(**data)
        
        return text_content, html_content

def create_notification_settings():
    """Create notification settings interface"""
    st.subheader("🔔 Notification Settings")
    
    with st.form("notification_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            email_notifications = st.checkbox("Email Notifications", value=True)
            submission_updates = st.checkbox("Submission Updates", value=True)
            document_alerts = st.checkbox("Document Alerts", value=True)
        
        with col2:
            approval_notifications = st.checkbox("Approval Status", value=True)
            deadline_reminders = st.checkbox("Deadline Reminders", value=True)
            monthly_reports = st.checkbox("Monthly Reports", value=False)
        
        notification_email = st.text_input(
            "Notification Email",
            value=st.session_state.get('institution_user', {}).get('email', ''),
            help="Email address to receive notifications"
        )
        
        frequency = st.selectbox(
            "Notification Frequency",
            ["Immediate", "Daily Digest", "Weekly Summary"],
            help="How often to receive notifications"
        )
        
        if st.form_submit_button("💾 Save Settings"):
            st.success("Notification settings saved!")
            
            # Save to database or session
            st.session_state.notification_settings = {
                "email": notification_email,
                "email_notifications": email_notifications,
                "frequency": frequency
            }
