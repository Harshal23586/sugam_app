def create_system_settings(analyzer):
    st.header("⚙️ System Configuration & Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Metrics", 
        "📋 Document Requirements",
        "🎯 Approval Thresholds",
        "🔧 System Parameters"
    ])
    
    with tab1:
        st.subheader("📊 Performance Metrics Configuration")
        st.info("Configure weights and parameters for institutional performance evaluation")
        
        # Create editable performance metrics structure
        performance_config = analyzer.performance_metrics.copy()
        
        for category, config in performance_config.items():
            with st.expander(f"🔸 {category.replace('_', ' ').title()} (Weight: {config['weight']})", expanded=True):
                
                # Category weight
                new_weight = st.slider(
                    f"Category Weight for {category.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=1.0,
                    value=config['weight'],
                    step=0.05,
                    key=f"weight_{category}"
                )
                
                config['weight'] = new_weight
                
                # Sub-metrics configuration
                st.write("**Sub-Metrics Configuration:**")
                sub_metrics = config['sub_metrics']
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    for sub_metric, weight in sub_metrics.items():
                        new_sub_weight = st.slider(
                            f"{sub_metric.replace('_', ' ').title()}",
                            min_value=0.0,
                            max_value=1.0,
                            value=weight,
                            step=0.05,
                            key=f"sub_{category}_{sub_metric}"
                        )
                        sub_metrics[sub_metric] = new_sub_weight
                
                with col2:
                    # Show weight distribution
                    total = sum(sub_metrics.values())
                    if total > 0:
                        st.write("**Distribution:**")
                        for sub_metric, weight in sub_metrics.items():
                            percentage = (weight / total) * 100
                            st.write(f"{sub_metric}: {percentage:.1f}%")
                    else:
                        st.warning("Total weight is 0")
        
        # Save button for performance metrics
        if st.button("💾 Save Performance Metrics Configuration", type="primary"):
            analyzer.performance_metrics = performance_config
            save_configuration(analyzer, 'performance_metrics', performance_config)
            st.success("✅ Performance metrics configuration saved!")
        
        # Reset to defaults
        if st.button("🔄 Reset to Defaults", type="secondary"):
            analyzer.performance_metrics = analyzer.define_performance_metrics()
            st.success("✅ Performance metrics reset to defaults")
            st.rerun()
    
    with tab2:
        st.subheader("📋 Document Requirements Configuration")
        st.info("Configure document requirements for different approval types")
        
        doc_config = analyzer.document_requirements.copy()
        
        for approval_type, requirements in doc_config.items():
            with st.expander(f"📄 {approval_type.replace('_', ' ').title()}", expanded=True):
                
                # Mandatory Documents
                st.write("**📌 Mandatory Documents:**")
                mandatory_docs = requirements['mandatory']
                
                # Dynamic list for mandatory documents
                new_mandatory = []
                for i, doc in enumerate(mandatory_docs):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        new_doc = st.text_input(
                            f"Mandatory Document {i+1}",
                            value=doc,
                            key=f"mandatory_{approval_type}_{i}"
                        )
                    with col2:
                        if st.button("❌", key=f"remove_mandatory_{approval_type}_{i}"):
                            continue  # Skip adding this one
                    if new_doc:
                        new_mandatory.append(new_doc)
                
                # Add new document button
                if st.button("➕ Add New Mandatory Document", key=f"add_mandatory_{approval_type}"):
                    new_mandatory.append("New Document")
                    st.rerun()
                
                requirements['mandatory'] = new_mandatory
                
                # Supporting Documents
                st.write("**📎 Supporting Documents:**")
                supporting_docs = requirements['supporting']
                
                new_supporting = []
                for i, doc in enumerate(supporting_docs):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        new_doc = st.text_input(
                            f"Supporting Document {i+1}",
                            value=doc,
                            key=f"supporting_{approval_type}_{i}"
                        )
                    with col2:
                        if st.button("❌", key=f"remove_supporting_{approval_type}_{i}"):
                            continue
                    if new_doc:
                        new_supporting.append(new_doc)
                
                if st.button("➕ Add New Supporting Document", key=f"add_supporting_{approval_type}"):
                    new_supporting.append("New Supporting Document")
                    st.rerun()
                
                requirements['supporting'] = new_supporting
        
        # Save button for document requirements
        if st.button("💾 Save Document Requirements", type="primary"):
            analyzer.document_requirements = doc_config
            save_configuration(analyzer, 'document_requirements', doc_config)
            st.success("✅ Document requirements saved!")
    
    with tab3:
        st.subheader("🎯 Approval Thresholds Configuration")
        st.info("Configure score thresholds for different approval levels")
        
        # Current thresholds (extracted from generate_approval_recommendation)
        thresholds = {
            "Full Approval (5 Years)": 8.0,
            "Provisional Approval (3 Years)": 7.0,
            "Conditional Approval (1 Year)": 6.0,
            "Strict Monitoring (1 Year)": 5.0,
            "Rejection": 0.0
        }
        
        updated_thresholds = {}
        
        for level, threshold in thresholds.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{level}**")
            with col2:
                new_threshold = st.number_input(
                    f"Threshold",
                    min_value=0.0,
                    max_value=10.0,
                    value=threshold,
                    step=0.5,
                    key=f"threshold_{level}"
                )
            updated_thresholds[level] = new_threshold
        
        # Risk level thresholds
        st.write("---")
        st.subheader("⚠️ Risk Level Thresholds")
        
        risk_thresholds = {
            "Low Risk": 8.0,
            "Medium Risk": 6.5,
            "High Risk": 5.0,
            "Critical Risk": 0.0
        }
        
        for level, threshold in risk_thresholds.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{level}**")
            with col2:
                new_threshold = st.number_input(
                    f"Risk Threshold",
                    min_value=0.0,
                    max_value=10.0,
                    value=threshold,
                    step=0.5,
                    key=f"risk_{level}"
                )
        
        if st.button("💾 Save Thresholds", type="primary"):
            save_configuration(analyzer, 'approval_thresholds', updated_thresholds)
            st.success("✅ Approval thresholds saved!")
    
    with tab4:
        st.subheader("🔧 System Parameters")
        st.info("Configure system-wide parameters and constants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data generation parameters
            st.write("**Data Generation:**")
            default_institutions = st.number_input(
                "Default Number of Institutions",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
            
            default_years = st.number_input(
                "Default Years of Data",
                min_value=1,
                max_value=20,
                value=10,
                step=1
            )
            
            # RAG settings
            st.write("**AI/RAG Settings:**")
            chunk_size = st.number_input(
                "Document Chunk Size",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50
            )
        
        with col2:
            # Scoring parameters
            st.write("**Scoring Parameters:**")
            min_performance_score = st.number_input(
                "Minimum Performance Score",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5
            )
            
            max_performance_score = st.number_input(
                "Maximum Performance Score",
                min_value=5.0,
                max_value=15.0,
                value=10.0,
                step=0.5
            )
            
            # Document validation
            st.write("**Document Validation:**")
            mandatory_threshold = st.slider(
                "Mandatory Document Threshold (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=5
            )
            
            review_period_days = st.number_input(
                "Review Period (days)",
                min_value=1,
                max_value=90,
                value=30,
                step=1
            )
        
        # System actions
        st.write("---")
        st.subheader("🛠️ System Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Clear All Cache", type="secondary"):
                clear_cache(analyzer)
        
        with col2:
            if st.button("📊 Recalculate All Scores", type="secondary"):
                recalculate_all_scores(analyzer)
        
        with col3:
            if st.button("🔍 Validate System Integrity", type="secondary"):
                validate_system_integrity(analyzer)
        
        # Save all system parameters
        if st.button("💾 Save System Parameters", type="primary"):
            system_params = {
                'default_institutions': default_institutions,
                'default_years': default_years,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'min_performance_score': min_performance_score,
                'max_performance_score': max_performance_score,
                'mandatory_threshold': mandatory_threshold,
                'review_period_days': review_period_days
            }
            save_configuration(analyzer, 'system_parameters', system_params)
            st.success("✅ System parameters saved!")

def save_configuration(analyzer, config_type, config_data):
    """Save configuration to database"""
    cursor = analyzer.conn.cursor()
    
    # Create configuration table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_configuration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_type TEXT,
            config_data TEXT,
            updated_by TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Save configuration
    cursor.execute('''
        INSERT INTO system_configuration (config_type, config_data, updated_by)
        VALUES (?, ?, ?)
    ''', (config_type, json.dumps(config_data), 'admin'))
    
    analyzer.conn.commit()

def clear_cache(analyzer):
    """Clear system cache"""
    # Clear session state
    for key in list(st.session_state.keys()):
        if key not in ['session_initialized', 'rag_initialized']:
            del st.session_state[key]
    
    # Clear temporary files
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('ugc_'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
    
    st.success("✅ System cache cleared!")
    st.rerun()

def recalculate_all_scores(analyzer):
    """Recalculate all performance scores based on new configuration"""
    with st.spinner("Recalculating all scores with new configuration..."):
        # Get current data
        current_data = analyzer.historical_data.copy()
        
        # Apply new scoring configuration
        for idx, row in current_data.iterrows():
            # Calculate score using new weights
            new_score = analyzer.calculate_performance_score({
                'naac_grade': row.get('naac_grade'),
                'nirf_ranking': row.get('nirf_ranking'),
                'student_faculty_ratio': row.get('student_faculty_ratio'),
                'phd_faculty_ratio': row.get('phd_faculty_ratio'),
                'placement_rate': row.get('placement_rate'),
                'research_publications': row.get('research_publications'),
                'digital_infrastructure': row.get('digital_infrastructure_score'),
                'financial_stability': row.get('financial_stability_score'),
                'community_engagement': row.get('community_projects')
            })
            
            current_data.at[idx, 'performance_score'] = new_score
            current_data.at[idx, 'approval_recommendation'] = analyzer.generate_approval_recommendation(new_score)
            current_data.at[idx, 'risk_level'] = analyzer.assess_risk_level(new_score)
        
        # Save updated data
        current_data.to_sql('institutions', analyzer.conn, if_exists='replace', index=False)
        analyzer.historical_data = current_data
        
        st.success("✅ All scores recalculated with new configuration!")

def validate_system_integrity(analyzer):
    """Validate system configuration integrity"""
    issues = []
    
    # Check if weights sum to 1
    total_weight = sum(category['weight'] for category in analyzer.performance_metrics.values())
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"Performance metric weights don't sum to 1.0 (current: {total_weight:.2f})")
    
    # Check document requirements
    for approval_type, requirements in analyzer.document_requirements.items():
        if not requirements.get('mandatory'):
            issues.append(f"No mandatory documents for {approval_type}")
    
    if issues:
        st.error("⚠️ System Integrity Issues Found:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("✅ System integrity validation passed!")
