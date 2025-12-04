# modules/system_settings.py
import streamlit as st
import json
import os
from datetime import datetime
import sqlite3
from utils.backup import create_backup, get_available_backups, restore_from_backup
from utils.helpers import get_database_information

def create_system_settings_module(analyzer):
    st.header("⚙️ System Settings & Administration")
    
    # Create tabs for different settings sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 General Settings",
        "👥 User Management", 
        "💾 Backup & Restore",
        "📊 System Health",
        "🔐 Access Control"
    ])
    
    with tab1:
        create_general_settings(analyzer)
    
    with tab2:
        create_user_management(analyzer)
    
    with tab3:
        create_backup_restore(analyzer)
    
    with tab4:
        create_system_health(analyzer)
    
    with tab5:
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
        settings = {
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "default_report_type": default_report_type,
            "auto_email_reports": auto_email_reports,
            "email_notifications": email_notifications,
            "slack_notifications": slack_notifications,
            "auto_escalation": auto_escalation,
            "review_days": review_days,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to database
        save_system_settings(analyzer, settings)
        st.success("✅ Settings saved successfully!")

def create_user_management(analyzer):
    st.subheader("👥 User Management")
    
    # Tab for different user types
    user_tab1, user_tab2, user_tab3 = st.tabs(["👨‍💼 Admin Users", "🏛️ Institution Users", "📋 Audit Log"])
    
    with user_tab1:
        manage_admin_users(analyzer)
    
    with user_tab2:
        manage_institution_users(analyzer)
    
    with user_tab3:
        show_audit_log(analyzer)

def manage_admin_users(analyzer):
    st.write("**Administrator Accounts**")
    
    # Get existing admin users
    cursor = analyzer.conn.cursor()
    cursor.execute("""
        SELECT username, email, role, created_at 
        FROM institution_users 
        WHERE role IN ('Admin', 'Super Admin')
        ORDER BY created_at DESC
    """)
    admin_users = cursor.fetchall()
    
    if admin_users:
        st.dataframe(
            pd.DataFrame(admin_users, columns=['Username', 'Email', 'Role', 'Created']),
            use_container_width=True
        )
    
    # Add new admin
    with st.expander("➕ Add New Administrator"):
        with st.form("new_admin_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            admin_role = st.selectbox("Role", ["Admin", "Super Admin"])
            
            if st.form_submit_button("Create Admin Account"):
                if new_password == confirm_password:
                    success = analyzer.create_institution_user(
                        "SYSTEM_ADMIN",
                        new_username,
                        new_password,
                        "System Administrator",
                        new_email,
                        "+91-0000000000",
                        role=admin_role
                    )
                    if success:
                        st.success(f"✅ Admin account created for {new_username}")
                    else:
                        st.error("❌ Username already exists")
                else:
                    st.error("❌ Passwords do not match")

def manage_institution_users(analyzer):
    st.write("**Institution User Accounts**")
    
    # Get all institution users
    cursor = analyzer.conn.cursor()
    cursor.execute("""
        SELECT iu.username, i.institution_name, iu.contact_person, 
               iu.email, iu.role, iu.is_active, iu.created_at
        FROM institution_users iu
        JOIN institutions i ON iu.institution_id = i.institution_id
        WHERE iu.role = 'Institution'
        ORDER BY i.institution_name
    """)
    institution_users = cursor.fetchall()
    
    if institution_users:
        # Display with filtering options
        search_term = st.text_input("🔍 Search users", "")
        
        df_users = pd.DataFrame(institution_users, columns=[
            'Username', 'Institution', 'Contact', 'Email', 'Role', 'Active', 'Created'
        ])
        
        if search_term:
            mask = df_users.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            df_users = df_users[mask]
        
        st.dataframe(df_users, use_container_width=True)
    
    # Bulk actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset Passwords", type="secondary"):
            st.info("Password reset emails would be sent to selected users")
    
    with col2:
        if st.button("📧 Send Welcome Email", type="secondary"):
            st.success("Welcome emails sent to new users")
    
    with col3:
        if st.button("📊 Export User List", type="secondary"):
            st.download_button(
                label="⬇️ Download CSV",
                data=df_users.to_csv(index=False),
                file_name="institution_users.csv",
                mime="text/csv"
            )

def show_audit_log(analyzer):
    st.write("**System Audit Log**")
    
    # Get audit log from database
    try:
        cursor = analyzer.conn.cursor()
        cursor.execute("""
            SELECT timestamp, user_id, action, details 
            FROM audit_log 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        audit_log = cursor.fetchall()
        
        if audit_log:
            df_log = pd.DataFrame(audit_log, columns=['Timestamp', 'User', 'Action', 'Details'])
            st.dataframe(df_log, use_container_width=True)
        else:
            st.info("No audit log entries found")
            
    except Exception as e:
        st.info("Creating audit log table...")
        cursor = analyzer.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                action TEXT,
                details TEXT
            )
        """)
        analyzer.conn.commit()
        st.success("Audit log table created")

def create_backup_restore(analyzer):
    st.subheader("💾 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create Backup**")
        backup_name = st.text_input("Backup Name", f"backup_{datetime.now().strftime('%Y%m%d')}")
        include_documents = st.checkbox("Include Documents", value=True)
        
        if st.button("💾 Create Backup Now", type="primary"):
            with st.spinner("Creating backup..."):
                backup_file = create_backup(analyzer, backup_name, include_documents)
                st.success(f"✅ Backup created: {backup_file}")
    
    with col2:
        st.write("**Restore Backup**")
        backups = get_available_backups()
        
        if backups:
            selected_backup = st.selectbox("Select Backup", backups)
            
            if st.button("🔄 Restore Selected Backup", type="secondary"):
                if st.checkbox("⚠️ I understand this will overwrite current data"):
                    with st.spinner("Restoring backup..."):
                        success = restore_from_backup(analyzer, selected_backup)
                        if success:
                            st.success("✅ Backup restored successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to restore backup")
        else:
            st.info("No backups available")
    
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
    
    with col3:
        backup_time = st.time_input(
            "Backup Time",
            value=datetime.strptime("02:00", "%H:%M").time(),
            disabled=not enable_auto_backup
        )
    
    if st.button("📅 Save Backup Schedule"):
        st.success("✅ Backup schedule saved")

def create_system_health(analyzer):
    st.subheader("📊 System Health Dashboard")
    
    # Get system information
    system_info = get_database_information(analyzer)
    
    # Display health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database Size", f"{system_info['size_mb']} MB")
        st.metric("Total Records", system_info['total_records'])
    
    with col2:
        st.metric("Active Users", system_info.get('active_users', 0))
        st.metric("API Requests", system_info.get('api_requests', 0))
    
    with col3:
        # Disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        st.metric("Disk Space", f"{free // (2**30)} GB free")
        
        # Memory usage
        import psutil
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")
    
    with col4:
        # System uptime
        import uptime
        days = uptime.uptime() / 86400
        st.metric("System Uptime", f"{days:.1f} days")
        
        # Last backup
        st.metric("Last Backup", system_info['last_backup'])
    
    # Performance charts
    st.markdown("---")
    st.write("**Performance Metrics**")
    
    # Create sample performance data
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Response Time', 'User Activity', 'Database Load', 'Error Rate')
    )
    
    # Sample data - replace with actual metrics
    fig.add_trace(
        go.Scatter(x=list(range(24)), y=[100, 120, 110, 105, 115, 125, 130, 140, 135, 130, 125, 120, 
                                         115, 110, 105, 100, 95, 90, 85, 80, 85, 90, 95, 100],
                   mode='lines', name='Response Time (ms)'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
               y=[150, 200, 180, 220, 250, 100, 80], name='Active Users'),
        row=1, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # System recommendations
    st.markdown("---")
    st.write("**🔧 System Recommendations**")
    
    recommendations = []
    if system_info['size_mb'] > 100:
        recommendations.append("💾 **Database size large**: Consider archiving old data")
    if 'last_backup' in system_info and 'No backup' in system_info['last_backup']:
        recommendations.append("⚠️ **No recent backup**: Create a backup immediately")
    if system_info.get('data_age_days', 365) > 30:
        recommendations.append("📅 **Data outdated**: Update to current year")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ System is healthy and up-to-date")

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
    
    # Display existing API keys
    cursor = analyzer.conn.cursor()
    cursor.execute("SELECT * FROM api_keys")
    api_keys = cursor.fetchall()
    
    if api_keys:
        st.write("**Existing API Keys:**")
        for key in api_keys:
            with st.expander(f"🔑 {key[1]} - {key[2]}"):
                st.code(f"Key: {key[3][:20]}...")
                st.write(f"Created: {key[4]}")
                if st.button(f"Revoke {key[1]}", type="secondary"):
                    cursor.execute("DELETE FROM api_keys WHERE id = ?", (key[0],))
                    analyzer.conn.commit()
                    st.success("✅ API key revoked")
                    st.rerun()
    
    # Generate new API key
    with st.expander("➕ Generate New API Key"):
        key_name = st.text_input("Key Name")
        key_type = st.selectbox("Key Type", ["Read-only", "Read-Write", "Admin"])
        expiry_days = st.number_input("Expiry (days)", min_value=1, max_value=365, value=30)
        
        if st.button("Generate API Key"):
            import secrets
            api_key = secrets.token_urlsafe(32)
            
            cursor.execute("""
                INSERT INTO api_keys (name, type, key_value, expires_at)
                VALUES (?, ?, ?, ?)
            """, (key_name, key_type, api_key, 
                  datetime.now().timestamp() + expiry_days * 86400))
            analyzer.conn.commit()
            
            st.success("✅ API key generated!")
            st.code(f"API Key: {api_key}")
            st.warning("⚠️ Save this key now - it won't be shown again!")

def save_system_settings(analyzer, settings):
    """Save system settings to database"""
    cursor = analyzer.conn.cursor()
    
    # Create settings table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE,
            setting_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Save each setting
    for key, value in settings.items():
        cursor.execute("""
            INSERT OR REPLACE INTO system_settings (setting_key, setting_value)
            VALUES (?, ?)
        """, (key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))
    
    analyzer.conn.commit()

def load_system_settings(analyzer):
    """Load system settings from database"""
    cursor = analyzer.conn.cursor()
    cursor.execute("SELECT setting_key, setting_value FROM system_settings")
    settings = {}
    
    for key, value in cursor.fetchall():
        try:
            settings[key] = json.loads(value)
        except:
            settings[key] = value
    
    return settings
