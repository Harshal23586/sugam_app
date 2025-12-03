import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
import glob
from datetime import datetime
import sqlite3

def create_data_management_module(analyzer):
    st.header("💾 Data Management & Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Current Data Analytics",
        "🔍 Data Validation & QA",
        "⚙️ Advanced Database Tools"
    ])
    
    with tab1:
        create_data_analytics_tab(analyzer)
    
    with tab2:
        create_data_validation_tab(analyzer)
    
    with tab3:
        create_database_tools_tab(analyzer)

def create_data_analytics_tab(analyzer):
    """Create data analytics tab"""
    st.subheader("📊 Current Database Analytics")
    
    current_data = analyzer.historical_data
    
    # Show database statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_records = len(current_data)
        st.metric("📊 Total Records", total_records)
    
    with col2:
        unique_institutions = current_data['institution_id'].nunique()
        st.metric("🏛️ Unique Institutions", unique_institutions)
    
    with col3:
        years_covered = current_data['year'].nunique()
        st.metric("📅 Years Covered", years_covered)
    
    with col4:
        year_range = f"{current_data['year'].min()}-{current_data['year'].max()}"
        st.metric("🗓️ Year Range", year_range)
    
    # Data quality indicators
    st.subheader("📈 Data Quality Indicators")
    
    current_year_data = current_data[current_data['year'] == 2023]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (current_year_data.count().sum() / current_year_data.size * 100)
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        naac_rated = current_year_data['naac_grade'].notna().sum()
        st.metric("NAAC Rated", f"{naac_rated}/{len(current_year_data)}")
    
    with col3:
        nirf_ranked = current_year_data['nirf_ranking'].notna().sum()
        st.metric("NIRF Ranked", f"{nirf_ranked}/{len(current_year_data)}")
    
    with col4:
        missing_values = current_year_data.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    # Search and filter
    st.subheader("🔍 Search & Filter Institutions")
    
    search_col1, search_col2 = st.columns(2)
    
    with search_col1:
        search_text = st.text_input("Search by Name/ID", "")
    
    with search_col2:
        institution_types = ["All"] + sorted(current_year_data['institution_type'].unique().tolist())
        selected_type = st.selectbox("Filter by Type", institution_types)
    
    # Apply filters
    filtered_data = current_year_data.copy()
    
    if search_text:
        mask = (filtered_data['institution_name'].str.contains(search_text, case=False)) | \
               (filtered_data['institution_id'].str.contains(search_text, case=False))
        filtered_data = filtered_data[mask]
    
    if selected_type != "All":
        filtered_data = filtered_data[filtered_data['institution_type'] == selected_type]
    
    st.info(f"**Showing {len(filtered_data)} of {len(current_year_data)} institutions**")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export to CSV", type="secondary"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="institutions_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.rerun()
    
    with col3:
        if st.button("📋 Generate Summary", type="secondary"):
            generate_data_summary(filtered_data)
    
    # Data preview
    st.subheader("📋 Data Preview")
    
    display_columns = ['institution_id', 'institution_name', 'institution_type', 'state', 
                      'performance_score', 'naac_grade', 'placement_rate', 'risk_level']
    
    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        height=400
    )

def create_data_validation_tab(analyzer):
    """Create data validation tab"""
    st.subheader("🔍 Data Quality Analysis & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Data Quality Checks**")
        if st.button("🔄 Run Data Validation", type="primary"):
            run_data_validation(analyzer)
        
        if st.button("📋 Show Data Completeness Report"):
            show_data_completeness_report(analyzer)
        
        if st.button("⚠️ Identify Data Anomalies"):
            identify_data_anomalies(analyzer)
    
    with col2:
        st.info("**Data Enhancement Tools**")
        if st.button("🎯 Fill Missing Values (AI)"):
            fill_missing_values_ai(analyzer)
        
        if st.button("📈 Calculate Derived Metrics"):
            calculate_derived_metrics(analyzer)

def create_database_tools_tab(analyzer):
    """Create database tools tab"""
    st.subheader("⚙️ Advanced Database Management")
    
    st.warning("⚠️ **Warning**: These operations affect the entire database. Use with caution.")
    
    # Backup & Restore
    st.markdown("### 📦 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Create Backup", type="secondary"):
            create_backup(analyzer)
    
    with col2:
        backup_files = get_available_backups()
        selected_backup = st.selectbox("Select Backup to Restore", backup_files)
        if st.button("🔄 Restore from Backup", type="secondary"):
            restore_from_backup(analyzer, selected_backup)
    
    # Data regeneration
    st.markdown("### 🔄 Data Regeneration")
    
    regenerate_option = st.selectbox("Regeneration Type", [
        "Regenerate Current Year Only",
        "Regenerate Performance Scores Only",
        "Regenerate All Data"
    ])
    
    if st.button("🔄 Regenerate Selected", type="secondary"):
        if st.checkbox("I understand this will overwrite existing data"):
            regenerate_data(analyzer, regenerate_option)
    
    # Maintenance operations
    st.markdown("### 🛠️ Maintenance Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Optimize Database", type="secondary"):
            optimize_database(analyzer)
    
    with col2:
        if st.button("🔍 Fix Data Inconsistencies", type="secondary"):
            fix_data_inconsistencies(analyzer)

# Helper functions for data management
def run_data_validation(analyzer):
    """Run comprehensive data validation"""
    with st.spinner("Running data validation checks..."):
        current_data = analyzer.historical_data
        current_year_data = current_data[current_data['year'] == 2023]
        
        issues = []
        warnings = []
        checks_passed = []
        
        # Check for missing values
        critical_columns = ['institution_id', 'institution_name', 'performance_score']
        for col in critical_columns:
            null_count = current_year_data[col].isnull().sum()
            if null_count > 0:
                issues.append(f"❌ Missing {col}: {null_count} institutions")
            else:
                checks_passed.append(f"✅ {col}: Complete")
        
        # Validate data ranges
        if 'placement_rate' in current_year_data.columns:
            invalid_placement = current_year_data[
                (current_year_data['placement_rate'] < 0) | 
                (current_year_data['placement_rate'] > 100)
            ].shape[0]
            if invalid_placement > 0:
                issues.append(f"❌ Invalid placement rates: {invalid_placement} records")
        
        # Display results
        if issues:
            st.error(f"**Critical Issues**: {len(issues)}")
            for issue in issues:
                st.write(issue)
        else:
            st.success("✅ No critical issues found")
        
        if checks_passed:
            st.success(f"**Checks Passed**: {len(checks_passed)}")
            for check in checks_passed[:5]:
                st.write(check)

def show_data_completeness_report(analyzer):
    """Generate data completeness report"""
    current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    
    completeness_by_column = {}
    for col in current_year_data.columns:
        non_null = current_year_data[col].notna().sum()
        total = len(current_year_data)
        completeness = (non_null / total) * 100 if total > 0 else 0
        
        completeness_by_column[col] = {
            'completeness': completeness,
            'missing': total - non_null
        }
    
    # Create visualization
    df_completeness = pd.DataFrame([
        {
            'Column': col,
            'Completeness %': info['completeness'],
            'Missing': info['missing']
        }
        for col, info in completeness_by_column.items()
    ]).sort_values('Completeness %')
    
    # Show top 10 most incomplete columns
    st.subheader("📊 Data Completeness Report")
    
    incomplete_cols = df_completeness[df_completeness['Completeness %'] < 100].head(10)
    
    if not incomplete_cols.empty:
        fig = px.bar(incomplete_cols,
                    x='Completeness %', y='Column',
                    orientation='h',
                    title="Top 10 Incomplete Columns",
                    color='Completeness %',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ All columns have 100% completeness!")

def identify_data_anomalies(analyzer):
    """Identify data anomalies"""
    current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    
    anomalies = []
    
    # Check for outliers in numeric columns
    numeric_cols = current_year_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col in ['institution_id', 'year', 'established_year']:
            continue
        
        data = current_year_data[col].dropna()
        if len(data) < 5:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = current_year_data[
            (current_year_data[col] < lower_bound) | 
            (current_year_data[col] > upper_bound)
        ]
        
        if len(outliers) > 0:
            for _, row in outliers.iterrows():
                anomalies.append({
                    'Institution': row['institution_name'],
                    'Column': col,
                    'Value': row[col],
                    'Type': 'Statistical Outlier'
                })
    
    if anomalies:
        df_anomalies = pd.DataFrame(anomalies)
        st.error(f"⚠️ Found {len(anomalies)} anomalies")
        st.dataframe(df_anomalies, use_container_width=True)
    else:
        st.success("✅ No statistical anomalies detected!")

def fill_missing_values_ai(analyzer):
    """Fill missing values using AI/statistical methods"""
    with st.spinner("Filling missing values..."):
        current_data = analyzer.historical_data.copy()
        
        # Simple imputation for demonstration
        for col in current_data.columns:
            if current_data[col].dtype == 'object':
                # For categorical, use mode
                mode_val = current_data[col].mode()
                if not mode_val.empty:
                    current_data[col].fillna(mode_val.iloc[0], inplace=True)
            else:
                # For numeric, use median
                current_data[col].fillna(current_data[col].median(), inplace=True)
        
        # Save back to database
        current_data.to_sql('institutions', analyzer.conn, if_exists='replace', index=False)
        analyzer.historical_data = current_data
        
        st.success("✅ Missing values filled using statistical methods!")

def calculate_derived_metrics(analyzer):
    """Calculate derived metrics"""
    with st.spinner("Calculating derived metrics..."):
        current_data = analyzer.historical_data.copy()
        
        # Calculate composite score
        if all(col in current_data.columns for col in ['performance_score', 'placement_rate', 'research_publications']):
            current_data['composite_score'] = (
                current_data['performance_score'] * 0.5 +
                (current_data['placement_rate'] / 10) * 0.3 +
                (current_data['research_publications'] / 50) * 0.2
            ).round(2)
        
        # Calculate growth rate
        current_data['growth_rate'] = current_data.groupby('institution_id')['performance_score'].pct_change() * 100
        
        # Save to database
        current_data.to_sql('institutions', analyzer.conn, if_exists='replace', index=False)
        analyzer.historical_data = current_data
        
        st.success("✅ Derived metrics calculated successfully!")

def create_backup(analyzer):
    """Create database backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = "data/backups"
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    backup_file = os.path.join(backup_dir, f"backup_{timestamp}.json")
    
    backup_data = {
        'metadata': {
            'backup_date': datetime.now().isoformat(),
            'total_records': len(analyzer.historical_data)
        },
        'data': analyzer.historical_data.to_dict('records')
    }
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    st.success(f"✅ Backup created: {backup_file}")

def get_available_backups():
    """Get list of available backups"""
    backup_dir = "data/backups"
    
    if not os.path.exists(backup_dir):
        return ["No backups available"]
    
    backup_files = glob.glob(os.path.join(backup_dir, "backup_*.json"))
    
    if not backup_files:
        return ["No backups available"]
    
    backup_names = []
    for file_path in backup_files:
        filename = os.path.basename(file_path)
        backup_names.append(filename)
    
    return backup_names

def restore_from_backup(analyzer, selected_backup):
    """Restore from backup"""
    st.info("Backup restoration functionality would be implemented here")

def regenerate_data(analyzer, option):
    """Regenerate data"""
    if option == "Regenerate Current Year Only":
        st.info("Current year data regeneration would be implemented here")
    elif option == "Regenerate Performance Scores Only":
        st.info("Performance scores regeneration would be implemented here")
    else:
        st.info("Full data regeneration would be implemented here")

def optimize_database(analyzer):
    """Optimize database"""
    with st.spinner("Optimizing database..."):
        cursor = analyzer.conn.cursor()
        cursor.execute("VACUUM")
        analyzer.conn.commit()
        st.success("✅ Database optimization complete!")

def fix_data_inconsistencies(analyzer):
    """Fix data inconsistencies"""
    with st.spinner("Fixing data inconsistencies..."):
        current_data = analyzer.historical_data.copy()
        
        # Fix risk level inconsistencies
        for idx, row in current_data.iterrows():
            score = row['performance_score']
            current_risk = row['risk_level']
            
            if score >= 8.0 and current_risk != 'Low Risk':
                current_data.at[idx, 'risk_level'] = 'Low Risk'
            elif 6.5 <= score < 8.0 and current_risk != 'Medium Risk':
                current_data.at[idx, 'risk_level'] = 'Medium Risk'
            elif score < 6.5 and current_risk not in ['High Risk', 'Critical Risk']:
                current_data.at[idx, 'risk_level'] = 'High Risk'
        
        # Save fixes
        current_data.to_sql('institutions', analyzer.conn, if_exists='replace', index=False)
        analyzer.historical_data = current_data
        
        st.success("✅ Data inconsistencies fixed!")

def generate_data_summary(data):
    """Generate data summary report"""
    with st.expander("📄 Data Summary Report", expanded=True):
        st.write("### 📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Institutions", len(data))
            st.metric("Avg Performance", f"{data['performance_score'].mean():.2f}")
        
        with col2:
            st.metric("Avg Placement Rate", f"{data['placement_rate'].mean():.1f}%")
            st.metric("Research Active", f"{len(data[data['research_publications'] > 20])}/{len(data)}")
        
        with col3:
            st.metric("NAAC A/A+ Rated", f"{len(data[data['naac_grade'].isin(['A', 'A+', 'A++'])]):.0f}")
            st.metric("Low Risk Institutions", f"{len(data[data['risk_level'] == 'Low Risk']):.0f}")
        
        # Export option
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Data",
            data=csv_data,
            file_name="institution_summary.csv",
            mime="text/csv"
        )
