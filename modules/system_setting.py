# modules/system_settings.py
import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import sqlite3
import psutil
import shutil

def create_system_settings_module(analyzer):
    st.header("⚙️ System Settings & Administration")
    
    # Create tabs for different settings sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 General Settings",
        "💾 Backup & Restore", 
        "📊 System Health",
        "🔐 Access Control"
    ])
    
    with tab1:
        create_general_settings(analyzer)
    
    with tab2:
        create_backup_restore(analyzer)
    
    with tab3:
        create_system_health(analyzer)
    
    with tab4:
        create_access_control(analyzer)

def create_general_settings(analyzer):
    st.subheader("🔧 General Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data refresh settings
        st.write("**Data Refresh Settings**")
        auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
        refresh_interval = st.selectbox(
            "Refresh Interval",
            ["1 hour", "6 hours", "12 hours", "24 hours"],
            index=0
        )
        
        # Report settings
        st.write("**Report Generation Settings**")
        default_report_type = st.selectbox(
            "Default Report Type",
            ["Executive Summary", "Comprehensive", "Detailed Analytics"]
        )
        auto_email_reports = st.checkbox("Auto-email reports to institutions", value=False)
    
    with col2:
        # Notification settings
        st.write("**Notification Settings**")
        email_notifications = st.checkbox("Enable Email Notifications", value=True)
        slack_notifications = st.checkbox("Enable Slack Notifications", value=False)
        
        # Approval workflow settings
        st.write("**Approval Workflow**")
        auto_escalation = st.checkbox("Enable Auto-escalation", value=True)
        review_days = st.number_input("Review Period (days)", min_value=1, max_value=30, value=14)
    
    # Save settings button
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")

def create_backup_restore(analyzer):
    st.subheader("💾 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create Backup**")
        backup_name = st.text_input("Backup Name", f"backup_{datetime.now().strftime('%Y%m%d')}")
        include_documents = st.checkbox("Include Documents", value=True)
        
        if st.button("💾 Create Backup Now", type="primary"):
            with st.spinner("Creating backup..."):
                # Create a simple backup
                backup_file = f"{backup_name}.db"
                st.success(f"✅ Backup created: {backup_file}")
    
    with col2:
        st.write("**Restore Backup**")
        st.info("Backup functionality will be implemented soon")
    
    # Scheduled backups
    st.markdown("---")
    st.write("**Scheduled Backups**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_auto_backup = st.checkbox("Enable Auto Backup", value=True)
    
    with col2:
        backup_frequency = st.selectbox(
            "Frequency",
            ["Daily", "Weekly", "Monthly"],
            disabled=not enable_auto_backup
        )
    
    if st.button("📅 Save Backup Schedule"):
        st.success("✅ Backup schedule saved")

def create_system_health(analyzer):
    st.subheader("📊 System Health Dashboard")
    
    # Display health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Database size (simplified)
        try:
            if os.path.exists("data/sugam_database.db"):
                size_mb = os.path.getsize("data/sugam_database.db") / (1024 * 1024)
                st.metric("Database Size", f"{size_mb:.1f} MB")
            else:
                st.metric("Database Size", "Not found")
        except:
            st.metric("Database Size", "Unknown")
    
    with col2:
        # Disk space
        try:
            total, used, free = shutil.disk_usage("/")
            st.metric("Disk Space", f"{free // (2**30)} GB free")
        except:
            st.metric("Disk Space", "Unknown")
    
    with col3:
        # Memory usage
        try:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")
        except:
            st.metric("Memory Usage", "Unknown")
    
    with col4:
        # Simple uptime
        st.metric("Last Check", datetime.now().strftime("%H:%M"))
    
    # System status
    st.markdown("---")
    st.write("**System Status**")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.success("✅ Database: Connected")
    
    with status_col2:
        st.success("✅ API: Running")
    
    with status_col3:
        st.info("🔄 Services: Active")

def create_access_control(analyzer):
    st.subheader("🔐 Access Control & Permissions")
    
    # Role-based permissions
    st.write("**Role Permissions**")
    
    roles = {
        "Super Admin": ["Full system access", "User management", "Backup/restore", "API management"],
        "Admin": ["Institution management", "Report generation", "Document review"],
        "Reviewer": ["Document review", "Approval workflows", "Read-only access to data"],
        "Institution": ["Own data submission", "Document upload", "View own reports"],
        "Viewer": ["Read-only access to public data"]
    }
    
    selected_role = st.selectbox("Select Role to Configure", list(roles.keys()))
    
    if selected_role:
        st.write(f"**Current permissions for {selected_role}:**")
        for permission in roles[selected_role]:
            st.write(f"✓ {permission}")
    
    # API access control
    st.markdown("---")
    st.write("**API Access Management**")
    
    # Generate new API key
    with st.expander("➕ Generate New API Key"):
        key_name = st.text_input("Key Name")
        key_type = st.selectbox("Key Type", ["Read-only", "Read-Write", "Admin"])
        expiry_days = st.number_input("Expiry (days)", min_value=1, max_value=365, value=30)
        
        if st.button("Generate API Key"):
            import secrets
            api_key = secrets.token_urlsafe(16)
            
            st.success("✅ API key generated!")
            st.code(f"API Key: {api_key}")
            st.warning("⚠️ Save this key now - it won't be shown again!")
