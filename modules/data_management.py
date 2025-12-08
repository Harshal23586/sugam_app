def create_data_analytics_tab(analyzer):
    """Create data analytics tab"""
    st.subheader("📊 Current Database Analytics")
    
    current_data = analyzer.historical_data
    
    # Show database statistics - FIXED: Show ALL data stats, not just current year
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
    
    # Data quality indicators - FIXED: Show for current year only in indicators
    st.subheader("📈 Data Quality Indicators (Current Year: 2023)")
    
    current_year_data = current_data[current_data['year'] == 2023]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (current_year_data.count().sum() / current_year_data.size * 100) if not current_year_data.empty else 0
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        naac_rated = current_year_data['naac_grade'].notna().sum() if not current_year_data.empty else 0
        st.metric("NAAC Rated", f"{naac_rated}/{len(current_year_data)}")
    
    with col3:
        nirf_ranked = current_year_data['nirf_ranking'].notna().sum() if not current_year_data.empty else 0
        st.metric("NIRF Ranked", f"{nirf_ranked}/{len(current_year_data)}")
    
    with col4:
        missing_values = current_year_data.isnull().sum().sum() if not current_year_data.empty else 0
        st.metric("Missing Values", missing_values)
    
    # Search and filter - FIXED: Allow filtering across all years or just current year
    st.subheader("🔍 Search & Filter Institutions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_text = st.text_input("Search by Name/ID", "")
    
    with col2:
        institution_types = ["All"] + sorted(current_data['institution_type'].unique().tolist())
        selected_type = st.selectbox("Filter by Type", institution_types)
    
    with col3:
        # ADDED: Year selector to filter by specific year or all years
        all_years = ["All Years"] + sorted(current_data['year'].unique().tolist())
        selected_year = st.selectbox("Filter by Year", all_years, index=0 if len(all_years) > 1 else 0)
    
    # Apply filters - FIXED: Apply filters to ALL data initially
    filtered_data = current_data.copy()
    
    # Filter by year if selected
    if selected_year != "All Years":
        filtered_data = filtered_data[filtered_data['year'] == selected_year]
    
    # Apply text search
    if search_text:
        mask = (filtered_data['institution_name'].str.contains(search_text, case=False, na=False)) | \
               (filtered_data['institution_id'].str.contains(search_text, case=False, na=False))
        filtered_data = filtered_data[mask]
    
    # Apply type filter
    if selected_type != "All":
        filtered_data = filtered_data[filtered_data['institution_type'] == selected_type]
    
    # Display which data is being shown
    if selected_year == "All Years":
        st.info(f"**Showing {len(filtered_data)} records across all years**")
        # Calculate unique institutions across years
        unique_institutions_in_filter = filtered_data['institution_id'].nunique()
        st.info(f"**Unique institutions: {unique_institutions_in_filter}**")
    else:
        st.info(f"**Showing {len(filtered_data)} records for year {selected_year}**")
    
    # Action buttons - FIXED: Export ALL filtered data, not just current year
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Filtered Data to CSV", type="secondary"):
            csv_data = filtered_data.to_csv(index=False)
            filename = f"institutions_data_{selected_year.replace(' ', '_') if selected_year != 'All Years' else 'all_years'}.csv"
            st.download_button(
                label=f"⬇️ Download CSV ({len(filtered_data)} records)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_filtered"
            )
    
    with col2:
        if st.button("📥 Export Complete 10-Year Dataset", type="primary"):
            csv_data = current_data.to_csv(index=False)
            st.download_button(
                label=f"⬇️ Download Full Dataset ({len(current_data)} records)",
                data=csv_data,
                file_name="complete_10_year_dataset.csv",
                mime="text/csv",
                key="download_complete"
            )
    
    with col3:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.rerun()
    
    # Data preview - FIXED: Show filtered data, not just current year
    st.subheader("📋 Data Preview")
    
    display_columns = ['year', 'institution_id', 'institution_name', 'institution_type', 'state', 
                      'performance_score', 'naac_grade', 'placement_rate', 'risk_level']
    
    # Filter to display columns that exist in the data
    available_columns = [col for col in display_columns if col in filtered_data.columns]
    
    if not filtered_data.empty:
        st.dataframe(
            filtered_data[available_columns],
            use_container_width=True,
            height=400
        )
    else:
        st.info("No data matches your filters. Try different search criteria.")
