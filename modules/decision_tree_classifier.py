# modules/decision_tree_classifier.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

class InstitutionalDecisionTreeClassifier:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.model = None
        self.label_encoder = LabelEncoder()
        self.features = None
        self.trained = False
        self.model_path = "models/decision_tree_model.pkl"
        
    def prepare_data(self):
        """Prepare data for decision tree training"""
        try:
            df = self.analyzer.historical_data.copy()
            
            # Select features for classification
            feature_columns = [
                'student_faculty_ratio',
                'phd_faculty_ratio',
                'research_publications',
                'research_grants_amount',
                'patents_filed',
                'industry_collaborations',
                'digital_infrastructure_score',
                'library_volumes',
                'laboratory_equipment_score',
                'financial_stability_score',
                'compliance_score',
                'administrative_efficiency',
                'placement_rate',
                'higher_education_rate',
                'entrepreneurship_cell_score',
                'community_projects',
                'rural_outreach_score',
                'inclusive_education_index',
                'performance_score'
            ]
            
            # Check which features exist in the dataframe
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Handle missing values
            df = df.dropna(subset=available_features + ['risk_level'])
            
            # Encode target variable
            df['risk_level_encoded'] = self.label_encoder.fit_transform(df['risk_level'])
            
            # Store features
            self.features = available_features
            
            return df[available_features], df['risk_level_encoded']
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def train_model(self, max_depth=None, min_samples_split=2):
        """Train decision tree model"""
        try:
            X, y = self.prepare_data()
            
            if X is None or y is None:
                st.error("Could not prepare data for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store metrics
            self.metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
            }
            
            self.trained = True
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def predict_risk(self, institution_data):
        """Predict risk level for new institution data"""
        try:
            if not self.trained:
                st.warning("Model not trained. Please train the model first.")
                return None
            
            # Convert to DataFrame if not already
            if not isinstance(institution_data, pd.DataFrame):
                institution_data = pd.DataFrame([institution_data])
            
            # Ensure all required features are present
            missing_features = [f for f in self.features if f not in institution_data.columns]
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return None
            
            # Select only the required features
            X_new = institution_data[self.features]
            
            # Predict
            predictions_encoded = self.model.predict(X_new)
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_new)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'predicted_risk': pred,
                    'confidence': max(prob),
                    'probabilities': dict(zip(self.label_encoder.classes_, prob))
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self):
        """Save trained model to file"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model and encoder
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'features': self.features,
                    'metrics': self.metrics
                }, f)
            
            return True
        except Exception as e:
            st.warning(f"Could not save model: {str(e)}")
            return False
    
    def load_model(self):
        """Load trained model from file"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.label_encoder = saved_data['label_encoder']
                self.features = saved_data['features']
                self.metrics = saved_data.get('metrics', {})
                self.trained = True
            
            return True
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}")
            return False
    
    def visualize_tree(self, max_depth=3):
        """Visualize decision tree"""
        try:
            if not self.trained:
                st.warning("Model not trained")
                return None
            
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                self.model,
                feature_names=self.features,
                class_names=self.label_encoder.classes_,
                filled=True,
                rounded=True,
                max_depth=max_depth,
                ax=ax
            )
            
            return fig
        except Exception as e:
            st.error(f"Error visualizing tree: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        if not self.trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def create_decision_tree_module(analyzer):
    """Create Streamlit interface for decision tree classification"""
    
    st.title("🌳 Decision Tree Classifier - Risk Level Prediction")
    st.markdown("---")
    
    # Initialize classifier
    classifier = InstitutionalDecisionTreeClassifier(analyzer)
    
    # Sidebar for model controls
    st.sidebar.header("Model Controls")
    
    # Try to load existing model
    if st.sidebar.button("📂 Load Existing Model"):
        if classifier.load_model():
            st.sidebar.success("Model loaded successfully!")
            st.success("✅ Model loaded from saved file!")
        else:
            st.sidebar.warning("No saved model found")
    
    st.sidebar.markdown("---")
    
    # Training parameters
    max_depth = st.sidebar.slider("Max Tree Depth", 3, 20, 5, 
                                 help="Maximum depth of the decision tree")
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2,
                                         help="Minimum number of samples required to split a node")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Train Model", 
        "📊 Model Performance", 
        "🔮 Predict New Institution",
        "📈 Visualizations"
    ])
    
    with tab1:
        st.header("Train Decision Tree Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Dataset Information:**
            - Target: Risk Level (Critical Risk, High Risk, Medium Risk)
            - Features: 19 performance metrics
            - Training uses 80% of data
            """)
        
        with col2:
            if st.button("🚀 Train New Model", type="primary", use_container_width=True):
                with st.spinner("Training decision tree model..."):
                    if classifier.train_model(max_depth=max_depth, 
                                             min_samples_split=min_samples_split):
                        st.success("✅ Model trained successfully!")
                        
                        # Show quick metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Accuracy", f"{classifier.metrics['accuracy']:.2%}")
                        with col_b:
                            st.metric("Features Used", len(classifier.features))
                        with col_c:
                            st.metric("Classes", len(classifier.label_encoder.classes_))
                    else:
                        st.error("❌ Failed to train model")
        
        # Show data preview
        with st.expander("📋 View Training Data Preview"):
            X, y = classifier.prepare_data()
            if X is not None and y is not None:
                preview_df = X.copy()
                preview_df['risk_level'] = classifier.label_encoder.inverse_transform(y)
                st.dataframe(preview_df.head(10))
                st.caption(f"Training data shape: {X.shape[0]} samples × {X.shape[1]} features")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        if not classifier.trained:
            st.warning("Please train or load a model first")
        else:
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{classifier.metrics['accuracy']:.2%}")
            
            with col2:
                st.metric("Feature Count", len(classifier.features))
            
            with col3:
                st.metric("Tree Depth", classifier.model.get_depth())
            
            # Classification report
            st.subheader("Classification Report")
            st.text(classifier.metrics['classification_report'])
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(classifier.metrics['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=classifier.label_encoder.classes_,
                       yticklabels=classifier.label_encoder.classes_,
                       ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = classifier.get_feature_importance()
            if importance_df is not None:
                # Bar chart
                fig = px.bar(importance_df.head(10), 
                            x='importance', y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features",
                            color='importance',
                            color_continuous_scale='viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full table
                with st.expander("View All Feature Importances"):
                    st.dataframe(importance_df)
    
    with tab3:
        st.header("Predict Risk for New Institution")
        
        if not classifier.trained:
            st.warning("Please train or load a model first")
        else:
            # Create input form for new institution
            st.subheader("Enter Institution Metrics")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                # Dynamically create input fields based on features
                input_data = {}
                
                # Group features by category for better organization
                academic_features = ['student_faculty_ratio', 'phd_faculty_ratio', 
                                   'higher_education_rate']
                research_features = ['research_publications', 'research_grants_amount', 
                                   'patents_filed', 'industry_collaborations']
                infra_features = ['digital_infrastructure_score', 'library_volumes', 
                                'laboratory_equipment_score']
                admin_features = ['financial_stability_score', 'compliance_score', 
                                'administrative_efficiency']
                placement_features = ['placement_rate', 'entrepreneurship_cell_score']
                outreach_features = ['community_projects', 'rural_outreach_score', 
                                   'inclusive_education_index']
                quality_features = ['performance_score']
                
                # Create tabs for different feature categories
                pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5, pred_tab6 = st.tabs([
                    "📚 Academic", "🔬 Research", "🏢 Infrastructure",
                    "📊 Administration", "💼 Placement", "🌍 Outreach"
                ])
                
                with pred_tab1:
                    for feature in academic_features:
                        if feature in classifier.features:
                            if feature == 'student_faculty_ratio':
                                input_data[feature] = st.slider(
                                    "Student-Faculty Ratio", 5.0, 40.0, 15.0, 0.1,
                                    help="Lower is better"
                                )
                            elif feature == 'phd_faculty_ratio':
                                input_data[feature] = st.slider(
                                    "PhD Faculty Ratio", 0.1, 1.0, 0.5, 0.01,
                                    help="Higher is better"
                                )
                            elif feature == 'higher_education_rate':
                                input_data[feature] = st.slider(
                                    "Higher Education Rate (%)", 5.0, 50.0, 20.0, 0.1,
                                    help="Percentage of students pursuing higher education"
                                )
                
                with pred_tab2:
                    for feature in research_features:
                        if feature in classifier.features:
                            if feature == 'research_publications':
                                input_data[feature] = st.number_input(
                                    "Research Publications", 0, 100, 20,
                                    help="Number of research publications"
                                )
                            elif feature == 'research_grants_amount':
                                input_data[feature] = st.number_input(
                                    "Research Grants Amount (₹)", 0, 1000000, 100000,
                                    step=10000,
                                    help="Total research grant amount in ₹"
                                )
                            elif feature == 'patents_filed':
                                input_data[feature] = st.number_input(
                                    "Patents Filed", 0, 20, 3,
                                    help="Number of patents filed"
                                )
                            elif feature == 'industry_collaborations':
                                input_data[feature] = st.number_input(
                                    "Industry Collaborations", 0, 20, 5,
                                    help="Number of industry collaborations"
                                )
                
                with pred_tab3:
                    for feature in infra_features:
                        if feature in classifier.features:
                            if feature == 'digital_infrastructure_score':
                                input_data[feature] = st.slider(
                                    "Digital Infrastructure Score", 1.0, 10.0, 6.0, 0.1,
                                    help="Score out of 10"
                                )
                            elif feature == 'library_volumes':
                                input_data[feature] = st.number_input(
                                    "Library Volumes", 1000, 50000, 10000, 1000,
                                    help="Number of books in library"
                                )
                            elif feature == 'laboratory_equipment_score':
                                input_data[feature] = st.slider(
                                    "Lab Equipment Score", 1.0, 10.0, 7.0, 0.1,
                                    help="Score out of 10"
                                )
                
                with pred_tab4:
                    for feature in admin_features:
                        if feature in classifier.features:
                            input_data[feature] = st.slider(
                                f"{feature.replace('_', ' ').title()}", 
                                1.0, 10.0, 7.0, 0.1,
                                help="Score out of 10"
                            )
                
                with pred_tab5:
                    for feature in placement_features:
                        if feature in classifier.features:
                            if feature == 'placement_rate':
                                input_data[feature] = st.slider(
                                    "Placement Rate (%)", 40.0, 100.0, 75.0, 0.1,
                                    help="Percentage of students placed"
                                )
                            elif feature == 'entrepreneurship_cell_score':
                                input_data[feature] = st.slider(
                                    "Entrepreneurship Cell Score", 1.0, 10.0, 6.0, 0.1,
                                    help="Score out of 10"
                                )
                
                with pred_tab6:
                    for feature in outreach_features + quality_features:
                        if feature in classifier.features:
                            if feature == 'performance_score':
                                input_data[feature] = st.slider(
                                    "Overall Performance Score", 1.0, 10.0, 5.5, 0.1,
                                    help="Overall performance score"
                                )
                            else:
                                input_data[feature] = st.slider(
                                    f"{feature.replace('_', ' ').title()}", 
                                    1.0, 10.0, 6.0, 0.1,
                                    help="Score out of 10"
                                )
                
                # Predict button
                predict_button = st.form_submit_button("🔮 Predict Risk Level", 
                                                     type="primary", 
                                                     use_container_width=True)
                
                if predict_button:
                    with st.spinner("Making prediction..."):
                        prediction = classifier.predict_risk(input_data)
                        
                        if prediction:
                            st.success("✅ Prediction complete!")
                            
                            # Display results in a nice format
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Risk level with color coding
                                risk_color = {
                                    'Critical Risk': 'red',
                                    'High Risk': 'orange',
                                    'Medium Risk': 'green',
                                    'Low Risk': 'blue'
                                }.get(prediction['predicted_risk'], 'gray')
                                
                                st.markdown(f"""
                                <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {risk_color};'>
                                <h3>Predicted Risk: <span style='color: {risk_color}'>{prediction['predicted_risk']}</span></h3>
                                <p>Confidence: <strong>{prediction['confidence']:.1%}</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Probability distribution
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(prediction['probabilities'].keys()),
                                        y=list(prediction['probabilities'].values()),
                                        marker_color=['red' if 'Critical' in k else 
                                                     'orange' if 'High' in k else 
                                                     'green' if 'Medium' in k else 'blue' 
                                                     for k in prediction['probabilities'].keys()]
                                    )
                                ])
                                fig.update_layout(
                                    title="Risk Probability Distribution",
                                    xaxis_title="Risk Level",
                                    yaxis_title="Probability",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col3:
                                # Key metrics affecting prediction
                                st.subheader("Key Factors")
                                # Get feature importance for this prediction
                                if hasattr(classifier.model, 'decision_path'):
                                    # This would show the decision path
                                    st.info("Decision tree path analysis available")
                                
                                # Show top 3 influential features
                                importance_df = classifier.get_feature_importance()
                                if importance_df is not None:
                                    top_features = importance_df.head(3)
                                    for _, row in top_features.iterrows():
                                        feature_value = input_data.get(row['feature'], 'N/A')
                                        st.write(f"**{row['feature']}**: {feature_value}")
    
    with tab4:
        st.header("Model Visualizations")
        
        if not classifier.trained:
            st.warning("Please train or load a model first")
        else:
            # Decision tree visualization
            st.subheader("Decision Tree Structure")
            viz_depth = st.slider("Visualization Depth", 2, 5, 3,
                                 help="Shallower trees are easier to visualize")
            
            fig = classifier.visualize_tree(max_depth=viz_depth)
            if fig:
                st.pyplot(fig)
                st.caption("Note: For full tree visualization, use depth > 5 in training")
            
            # Feature importance sunburst
            st.subheader("Interactive Feature Importance")
            importance_df = classifier.get_feature_importance()
            if importance_df is not None:
                # Group features by category
                def categorize_feature(feature):
                    if any(f in feature for f in ['student', 'phd', 'education']):
                        return 'Academic'
                    elif any(f in feature for f in ['research', 'patent', 'industry']):
                        return 'Research'
                    elif any(f in feature for f in ['infrastructure', 'library', 'laboratory']):
                        return 'Infrastructure'
                    elif any(f in feature for f in ['financial', 'compliance', 'administrative']):
                        return 'Administration'
                    elif any(f in feature for f in ['placement', 'entrepreneurship']):
                        return 'Placement'
                    elif any(f in feature for f in ['community', 'rural', 'inclusive']):
                        return 'Outreach'
                    elif 'performance' in feature:
                        return 'Quality'
                    else:
                        return 'Other'
                
                importance_df['category'] = importance_df['feature'].apply(categorize_feature)
                
                # Create sunburst chart
                fig = px.sunburst(
                    importance_df,
                    path=['category', 'feature'],
                    values='importance',
                    color='importance',
                    color_continuous_scale='RdYlGn',
                    title="Feature Importance Hierarchy"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison suggestion
            st.subheader("Model Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Decision Tree Advantages:**
                - Easy to interpret and visualize
                - No feature scaling needed
                - Handles both numerical and categorical data
                - Non-parametric model
                """)
            
            with col2:
                st.warning("""
                **Considerations:**
                - Can overfit without proper depth control
                - Sensitive to small data changes
                - Biased towards features with many levels
                - Consider using Random Forest for better generalization
                """)
    
    # Add export functionality
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    
    if classifier.trained:
        if st.sidebar.button("💾 Save Model"):
            if classifier.save_model():
                st.sidebar.success("Model saved!")
            else:
                st.sidebar.warning("Could not save model")
        
        # Export predictions
        if st.sidebar.button("📊 Export Predictions"):
            # Generate predictions for all institutions
            df = analyzer.historical_data.copy()
            predictions = []
            
            for idx, row in df.iterrows():
                prediction = classifier.predict_risk(row)
                if prediction:
                    row_dict = row.to_dict()
                    row_dict['predicted_risk'] = prediction['predicted_risk']
                    row_dict['prediction_confidence'] = prediction['confidence']
                    predictions.append(row_dict)
            
            if predictions:
                pred_df = pd.DataFrame(predictions)
                csv = pred_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="institution_risk_predictions.csv",
                    mime="text/csv"
                )
