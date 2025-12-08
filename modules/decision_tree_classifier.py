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
import tempfile

class InstitutionalDecisionTreeClassifier:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.model = None
        self.label_encoder = LabelEncoder()
        self.features = None
        
        # Initialize session state for model persistence
        if 'dt_classifier_initialized' not in st.session_state:
            st.session_state.dt_classifier_initialized = False
            st.session_state.dt_model = None
            st.session_state.dt_label_encoder = None
            st.session_state.dt_features = None
            st.session_state.dt_metrics = None
            st.session_state.dt_trained = False
        
        # Load from session state if available
        if st.session_state.dt_trained:
            self.model = st.session_state.dt_model
            self.label_encoder = st.session_state.dt_label_encoder
            self.features = st.session_state.dt_features
            self.metrics = st.session_state.dt_metrics
        
        self.model_path = "models/decision_tree_model.pkl"
        
    @property
    def trained(self):
        """Check if model is trained (from session state)"""
        return st.session_state.get('dt_trained', False)
    
    @trained.setter
    def trained(self, value):
        """Set trained status in session state"""
        st.session_state.dt_trained = value
    
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
            
            # Display data info
            st.info(f"📊 Training on {X.shape[0]} samples with {X.shape[1]} features")
            
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
                                                             target_names=self.label_encoder.classes_,
                                                             output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
                'classes': self.label_encoder.classes_.tolist(),
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            # Save to session state
            st.session_state.dt_model = self.model
            st.session_state.dt_label_encoder = self.label_encoder
            st.session_state.dt_features = self.features
            st.session_state.dt_metrics = self.metrics
            st.session_state.dt_trained = True
            st.session_state.dt_classifier_initialized = True
            
            # Save model to file
            self.save_model()
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def predict_risk(self, institution_data):
        """Predict risk level for new institution data"""
        try:
            if not self.trained:
                # Try to load from session state
                if st.session_state.dt_trained:
                    self.model = st.session_state.dt_model
                    self.label_encoder = st.session_state.dt_label_encoder
                    self.features = st.session_state.dt_features
                else:
                    # Try to load from file
                    if not self.load_model():
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
                    'probabilities': dict(zip(self.label_encoder.classes_, prob)),
                    'all_probabilities': prob,
                    'prediction_encoded': predictions_encoded[i]
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
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
                # Check if model is in session state
                if st.session_state.dt_trained:
                    self.model = st.session_state.dt_model
                    self.label_encoder = st.session_state.dt_label_encoder
                    self.features = st.session_state.dt_features
                    self.metrics = st.session_state.dt_metrics
                    return True
                return False
            
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.label_encoder = saved_data['label_encoder']
                self.features = saved_data['features']
                self.metrics = saved_data.get('metrics', {})
                
                # Update session state
                st.session_state.dt_model = self.model
                st.session_state.dt_label_encoder = self.label_encoder
                st.session_state.dt_features = self.features
                st.session_state.dt_metrics = self.metrics
                st.session_state.dt_trained = True
            
            return True
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}")
            # Try to load from session state
            if st.session_state.dt_trained:
                self.model = st.session_state.dt_model
                self.label_encoder = st.session_state.dt_label_encoder
                self.features = st.session_state.dt_features
                self.metrics = st.session_state.dt_metrics
                return True
            return False
    
    def visualize_tree(self, max_depth=3):
        """Visualize decision tree"""
        try:
            if not self.trained:
                if not self.load_model():
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
                ax=ax,
                fontsize=10
            )
            
            return fig
        except Exception as e:
            st.error(f"Error visualizing tree: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        if not self.trained:
            if not self.load_model():
                return None
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model_info(self):
        """Get model information"""
        if not self.trained:
            return None
        
        info = {
            'trained': self.trained,
            'model_type': 'DecisionTreeClassifier',
            'feature_count': len(self.features) if self.features else 0,
            'classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
            'tree_depth': self.model.get_depth() if self.model else None,
            'accuracy': self.metrics.get('accuracy', 0) if self.metrics else 0
        }
        
        return info

def create_decision_tree_module(analyzer):
    """Create Streamlit interface for decision tree classification"""
    
    st.title("🌳 Decision Tree Classifier - Risk Level Prediction")
    st.markdown("---")
    
    # Initialize classifier
    classifier = InstitutionalDecisionTreeClassifier(analyzer)
    
    # Display model status
    model_info = classifier.get_model_info()
    if model_info and model_info['trained']:
        st.success(f"✅ Model loaded: {len(model_info['features'])} features, "
                  f"{model_info['accuracy']:.1%} accuracy")
    else:
        st.warning("⚠️ No trained model found. Please train a model first.")
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🛠️ Model Controls")
        
        # Model status indicator
        if classifier.trained:
            st.success("✅ Model is trained")
            if model_info:
                st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
                st.metric("Features", model_info['feature_count'])
                st.metric("Tree Depth", model_info['tree_depth'])
        else:
            st.warning("⚠️ Model not trained")
        
        st.markdown("---")
        
        # Load existing model
        if st.button("📂 Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                if classifier.load_model():
                    st.success("Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("No saved model found")
        
        # Training parameters
        st.markdown("---")
        st.subheader("Training Parameters")
        max_depth = st.slider("Max Tree Depth", 3, 20, 5, 
                             help="Maximum depth of the decision tree")
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2,
                                     help="Minimum number of samples required to split a node")
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training decision tree model..."):
                if classifier.train_model(max_depth=max_depth, 
                                         min_samples_split=min_samples_split):
                    st.success("✅ Model trained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to train model")
        
        # Save model button
        st.markdown("---")
        if classifier.trained:
            if st.button("💾 Save Model", use_container_width=True):
                if classifier.save_model():
                    st.success("Model saved!")
                else:
                    st.warning("Could not save model")
            
            # Export predictions
            if st.button("📊 Export Predictions", use_container_width=True):
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
                    
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="institution_risk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Performance", 
        "🔮 Predict New Institution",
        "📈 Visualizations"
    ])
    
    with tab1:
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
            
            # Data split info
            st.subheader("Data Split")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", classifier.metrics.get('training_samples', 0) + 
                         classifier.metrics.get('test_samples', 0))
            with col2:
                st.metric("Training Samples", classifier.metrics.get('training_samples', 0))
            with col3:
                st.metric("Test Samples", classifier.metrics.get('test_samples', 0))
            
            # Classification report
            st.subheader("Classification Report")
            if 'classification_report' in classifier.metrics:
                report_df = pd.DataFrame(classifier.metrics['classification_report']).transpose()
                st.dataframe(report_df.style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:,.0f}'
                }))
            
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
            ax.set_title('Confusion Matrix')
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
                    st.dataframe(importance_df.style.format({'importance': '{:.4f}'}))
    
    with tab2:
        st.header("Predict Risk for New Institution")
        
        # Always show model status at the top of the tab
        if not classifier.trained:
            st.error("❌ Model not trained or loaded!")
            st.info("Please go to the sidebar and click '🚀 Train Model' or '📂 Load Model' first.")
            return
        
        st.success("✅ Model is ready for predictions!")
        
        # Create input form for new institution
        st.subheader("Enter Institution Metrics")
        
        # Create a form with organized sections
        with st.form("prediction_form"):
            # Academic Metrics Section
            st.markdown("### 📚 Academic Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                student_faculty_ratio = st.slider(
                    "Student-Faculty Ratio", 5.0, 40.0, 15.0, 0.1,
                    help="Lower is better"
                )
            
            with col2:
                phd_faculty_ratio = st.slider(
                    "PhD Faculty Ratio", 0.1, 1.0, 0.5, 0.01,
                    help="Higher is better"
                )
            
            with col3:
                higher_education_rate = st.slider(
                    "Higher Education Rate (%)", 5.0, 50.0, 20.0, 0.1,
                    help="Percentage of students pursuing higher education"
                )
            
            # Research Metrics Section
            st.markdown("### 🔬 Research Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                research_publications = st.number_input(
                    "Research Publications", 0, 100, 20,
                    help="Number of research publications"
                )
            
            with col2:
                research_grants_amount = st.number_input(
                    "Research Grants (₹)", 0, 1000000, 100000,
                    step=10000,
                    help="Total research grant amount in ₹"
                )
            
            with col3:
                patents_filed = st.number_input(
                    "Patents Filed", 0, 20, 3,
                    help="Number of patents filed"
                )
            
            with col4:
                industry_collaborations = st.number_input(
                    "Industry Collaborations", 0, 20, 5,
                    help="Number of industry collaborations"
                )
            
            # Infrastructure Metrics Section
            st.markdown("### 🏢 Infrastructure Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                digital_infrastructure_score = st.slider(
                    "Digital Infrastructure Score", 1.0, 10.0, 6.0, 0.1,
                    help="Score out of 10"
                )
            
            with col2:
                library_volumes = st.number_input(
                    "Library Volumes", 1000, 50000, 10000, 1000,
                    help="Number of books in library"
                )
            
            with col3:
                laboratory_equipment_score = st.slider(
                    "Lab Equipment Score", 1.0, 10.0, 7.0, 0.1,
                    help="Score out of 10"
                )
            
            # Administrative Metrics Section
            st.markdown("### 📊 Administrative Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                financial_stability_score = st.slider(
                    "Financial Stability Score", 1.0, 10.0, 7.0, 0.1,
                    help="Score out of 10"
                )
            
            with col2:
                compliance_score = st.slider(
                    "Compliance Score", 1.0, 10.0, 7.0, 0.1,
                    help="Score out of 10"
                )
            
            with col3:
                administrative_efficiency = st.slider(
                    "Administrative Efficiency", 1.0, 10.0, 6.5, 0.1,
                    help="Score out of 10"
                )
            
            # Placement Metrics Section
            st.markdown("### 💼 Placement Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                placement_rate = st.slider(
                    "Placement Rate (%)", 40.0, 100.0, 75.0, 0.1,
                    help="Percentage of students placed"
                )
            
            with col2:
                entrepreneurship_cell_score = st.slider(
                    "Entrepreneurship Cell Score", 1.0, 10.0, 6.0, 0.1,
                    help="Score out of 10"
                )
            
            # Outreach and Quality Metrics Section
            st.markdown("### 🌍 Outreach & Quality Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                community_projects = st.number_input(
                    "Community Projects", 0, 20, 5,
                    help="Number of community projects"
                )
            
            with col2:
                rural_outreach_score = st.slider(
                    "Rural Outreach Score", 1.0, 10.0, 6.0, 0.1,
                    help="Score out of 10"
                )
            
            with col3:
                inclusive_education_index = st.slider(
                    "Inclusive Education Index", 1.0, 10.0, 6.5, 0.1,
                    help="Score out of 10"
                )
            
            # Overall Performance
            st.markdown("### 🏆 Overall Performance")
            performance_score = st.slider(
                "Overall Performance Score", 1.0, 10.0, 5.5, 0.1,
                help="Overall performance score"
            )
            
            # Predict button
            predict_button = st.form_submit_button("🔮 Predict Risk Level", 
                                                 type="primary", 
                                                 use_container_width=True)
            
            if predict_button:
                # Prepare input data
                input_data = {
                    'student_faculty_ratio': student_faculty_ratio,
                    'phd_faculty_ratio': phd_faculty_ratio,
                    'research_publications': research_publications,
                    'research_grants_amount': research_grants_amount,
                    'patents_filed': patents_filed,
                    'industry_collaborations': industry_collaborations,
                    'digital_infrastructure_score': digital_infrastructure_score,
                    'library_volumes': library_volumes,
                    'laboratory_equipment_score': laboratory_equipment_score,
                    'financial_stability_score': financial_stability_score,
                    'compliance_score': compliance_score,
                    'administrative_efficiency': administrative_efficiency,
                    'placement_rate': placement_rate,
                    'higher_education_rate': higher_education_rate,
                    'entrepreneurship_cell_score': entrepreneurship_cell_score,
                    'community_projects': community_projects,
                    'rural_outreach_score': rural_outreach_score,
                    'inclusive_education_index': inclusive_education_index,
                    'performance_score': performance_score
                }
                
                with st.spinner("Analyzing institution data..."):
                    prediction = classifier.predict_risk(input_data)
                    
                    if prediction:
                        st.success("✅ Prediction complete!")
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        # Risk level with color coding
                        risk_level = prediction['predicted_risk']
                        confidence = prediction['confidence']
                        
                        if 'Critical' in risk_level:
                            color = '#dc3545'
                            icon = '🔴'
                            emoji = '⛔'
                        elif 'High' in risk_level:
                            color = '#fd7e14'
                            icon = '🟠'
                            emoji = '⚠️'
                        elif 'Medium' in risk_level:
                            color = '#ffc107'
                            icon = '🟡'
                            emoji = '📊'
                        else:
                            color = '#28a745'
                            icon = '🟢'
                            emoji = '✅'
                        
                        # Main result card
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {color};'>
                            <h2 style='color: {color}; margin-bottom: 10px;'>{emoji} {risk_level}</h2>
                            <p style='font-size: 16px;'><strong>Confidence:</strong> {confidence:.1%}</p>
                            <p style='font-size: 14px; color: #6c757d;'>Based on analysis of 19 performance metrics</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Quick recommendations
                            recommendations = {
                                'Critical Risk': "Immediate improvements required. Consider external audit.",
                                'High Risk': "Close monitoring needed. Submit improvement plan within 30 days.",
                                'Medium Risk': "Regular monitoring recommended. Focus on weak areas.",
                                'Low Risk': "Good performance. Continue current practices."
                            }
                            
                            recommendation = recommendations.get(risk_level, "Review all metrics for improvement.")
                            st.info(f"**Recommendation:** {recommendation}")
                        
                        # Probability distribution
                        st.subheader("📈 Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Risk Level': list(prediction['probabilities'].keys()),
                            'Probability': list(prediction['probabilities'].values())
                        })
                        
                        # Create bar chart
                        fig = px.bar(prob_df, 
                                    x='Risk Level', 
                                    y='Probability',
                                    color='Probability',
                                    color_continuous_scale='RdYlGn_r',
                                    title="Risk Probability Distribution")
                        fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show probabilities table
                        with st.expander("View Detailed Probabilities"):
                            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                        
                        # Feature analysis
                        st.subheader("🔍 Key Influencing Factors")
                        importance_df = classifier.get_feature_importance()
                        if importance_df is not None:
                            # Get top 5 features
                            top_features = importance_df.head(5)
                            
                            for _, row in top_features.iterrows():
                                feature_name = row['feature'].replace('_', ' ').title()
                                feature_value = input_data.get(row['feature'], 'N/A')
                                importance = row['importance']
                                
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.write(f"**{feature_name}**")
                                with col2:
                                    st.write(f"Value: {feature_value}")
                                with col3:
                                    st.progress(float(importance), text=f"Impact: {importance:.1%}")
    
    with tab3:
        st.header("Model Visualizations")
        
        if not classifier.trained:
            st.warning("Please train or load a model first")
        else:
            # Decision tree visualization
            st.subheader("Decision Tree Structure")
            viz_depth = st.slider("Visualization Depth", 2, 5, 3,
                                 help="Shallower trees are easier to visualize",
                                 key="viz_depth")
            
            fig = classifier.visualize_tree(max_depth=viz_depth)
            if fig:
                st.pyplot(fig)
                st.caption("Note: Full tree visualization might be complex. Consider using depth 3-4 for clarity.")
            
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
            st.subheader("📝 Model Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **✅ Decision Tree Advantages:**
                - Easy to interpret and visualize
                - No feature scaling needed
                - Handles both numerical and categorical data
                - Non-parametric model
                - Shows decision rules clearly
                """)
            
            with col2:
                st.warning("""
                **⚠️ Considerations:**
                - Can overfit without proper depth control
                - Sensitive to small data changes
                - Biased towards features with many levels
                - Consider Random Forest for better generalization
                - Regular pruning may be needed
                """)
