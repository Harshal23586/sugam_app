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
        
        # Initialize model components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.features = None
        self.metrics = {}
        
        # Initialize session state for persistence
        if 'dt_model_trained' not in st.session_state:
            st.session_state.dt_model_trained = False
        
        # Try to load from session state
        if 'dt_model' in st.session_state:
            self.model = st.session_state.dt_model
        if 'dt_label_encoder' in st.session_state:
            self.label_encoder = st.session_state.dt_label_encoder
        if 'dt_features' in st.session_state:
            self.features = st.session_state.dt_features
        if 'dt_metrics' in st.session_state:
            self.metrics = st.session_state.dt_metrics
        
        self.model_path = "models/decision_tree_model.pkl"
        
    @property
    def trained(self):
        """Check if model is trained"""
        return st.session_state.get('dt_model_trained', False)
    
    def prepare_data(self):
        """Prepare data for decision tree training"""
        try:
            df = self.analyzer.historical_data.copy()
            
            # Display data info
            st.write(f"📊 Raw data shape: {df.shape}")
            st.write(f"📊 Risk levels in data: {df['risk_level'].unique()}")
            
            # Define all possible features
            all_features = [
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
            
            # Check which features actually exist in the data
            available_features = []
            for feature in all_features:
                if feature in df.columns:
                    # Check if feature has non-null values
                    if df[feature].notna().sum() > 0:
                        available_features.append(feature)
                    else:
                        st.warning(f"Feature '{feature}' has no valid values")
                else:
                    st.warning(f"Feature '{feature}' not found in data columns")
            
            st.write(f"✅ Available features: {len(available_features)}")
            st.write(f"📋 Features list: {available_features}")
            
            # Remove rows with missing values in target
            df = df.dropna(subset=['risk_level'])
            
            # Handle missing values in features
            df_clean = df.copy()
            missing_counts = {}
            for feature in available_features:
                missing = df_clean[feature].isna().sum()
                if missing > 0:
                    missing_counts[feature] = missing
                    # Fill with median for numerical features
                    df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
            
            if missing_counts:
                st.warning(f"Missing values filled: {missing_counts}")
            
            # Encode target variable
            df_clean['risk_level_encoded'] = self.label_encoder.fit_transform(df_clean['risk_level'])
            
            # Store features
            self.features = available_features
            
            st.write(f"📊 Clean data shape: {df_clean.shape}")
            st.write(f"🎯 Classes: {self.label_encoder.classes_}")
            st.write(f"📈 Class distribution: {df_clean['risk_level'].value_counts().to_dict()}")
            
            return df_clean[available_features], df_clean['risk_level_encoded']
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def train_model(self, max_depth=None, min_samples_split=2):
        """Train decision tree model"""
        try:
            X, y = self.prepare_data()
            
            if X is None or y is None:
                st.error("Could not prepare data for training")
                return False
            
            if len(X) == 0:
                st.error("No data available for training")
                return False
            
            st.success(f"✅ Data prepared: {X.shape[0]} samples × {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.write(f"📚 Training samples: {X_train.shape[0]}")
            st.write(f"🧪 Test samples: {X_test.shape[0]}")
            
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
                'test_samples': X_test.shape[0],
                'feature_count': X.shape[1]
            }
            
            # Save to session state
            st.session_state.dt_model = self.model
            st.session_state.dt_label_encoder = self.label_encoder
            st.session_state.dt_features = self.features
            st.session_state.dt_metrics = self.metrics
            st.session_state.dt_model_trained = True
            
            st.success("✅ Model trained and saved to session state!")
            
            # Save model to file
            self.save_model()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def predict_risk(self, institution_data):
        """Predict risk level for new institution data"""
        try:
            # Check if model is loaded
            if not self.trained:
                # Try to load from session state
                if 'dt_model' in st.session_state and st.session_state.dt_model_trained:
                    self.model = st.session_state.dt_model
                    self.label_encoder = st.session_state.dt_label_encoder
                    self.features = st.session_state.dt_features
                    self.metrics = st.session_state.dt_metrics
                    st.session_state.dt_model_trained = True
                else:
                    # Try to load from file
                    if not self.load_model():
                        st.warning("⚠️ Model not trained. Please train the model first.")
                        return None
            
            if self.model is None:
                st.error("❌ Model is not initialized")
                return None
            
            # Convert to DataFrame if not already
            if not isinstance(institution_data, pd.DataFrame):
                institution_data = pd.DataFrame([institution_data])
            
            # Ensure all required features are present
            if self.features is None:
                st.error("❌ Features not defined")
                return None
            
            missing_features = [f for f in self.features if f not in institution_data.columns]
            if missing_features:
                st.error(f"❌ Missing features: {missing_features}")
                st.info(f"📋 Required features: {self.features}")
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
            st.error(f"❌ Error making prediction: {str(e)}")
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
            
            st.info(f"💾 Model saved to: {self.model_path}")
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not save model: {str(e)}")
            return False
    
    def load_model(self):
        """Load trained model from file"""
        try:
            if not os.path.exists(self.model_path):
                st.warning(f"⚠️ Model file not found: {self.model_path}")
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
                st.session_state.dt_model_trained = True
            
            st.success(f"✅ Model loaded from: {self.model_path}")
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not load model: {str(e)}")
            return False
    
    def visualize_tree(self, max_depth=3):
        """Visualize decision tree"""
        try:
            if not self.trained:
                st.warning("⚠️ Model not trained")
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
            st.error(f"❌ Error visualizing tree: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        if not self.trained:
            return None
        
        if self.model is None or self.features is None:
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
            'accuracy': self.metrics.get('accuracy', 0) if self.metrics else 0,
            'training_samples': self.metrics.get('training_samples', 0),
            'test_samples': self.metrics.get('test_samples', 0)
        }
        
        return info

def create_decision_tree_module(analyzer):
    """Create Streamlit interface for decision tree classification"""
    
    st.title("🌳 Decision Tree Classifier - Risk Level Prediction")
    st.markdown("---")
    
    # Initialize classifier
    classifier = InstitutionalDecisionTreeClassifier(analyzer)
    
    # Display data info
    st.subheader("📊 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(analyzer.historical_data))
    with col2:
        st.metric("Unique Institutions", analyzer.historical_data['institution_id'].nunique())
    with col3:
        st.metric("Data Years", f"{analyzer.historical_data['year'].min()}-{analyzer.historical_data['year'].max()}")
    
    # Check if model is trained
    if classifier.trained:
        st.success("✅ Model is trained and ready!")
        model_info = classifier.get_model_info()
        if model_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
            with col2:
                st.metric("Features", model_info['feature_count'])
            with col3:
                st.metric("Tree Depth", model_info['tree_depth'])
    else:
        st.warning("⚠️ Model not trained yet. Please train the model first.")
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🛠️ Model Controls")
        
        # Training parameters
        st.subheader("Training Parameters")
        max_depth = st.slider("Max Tree Depth", 3, 20, 5, 
                             help="Maximum depth of the decision tree")
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2,
                                     help="Minimum number of samples required to split a node")
        
        # Train button
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training decision tree model..."):
                if classifier.train_model(max_depth=max_depth, 
                                         min_samples_split=min_samples_split):
                    st.success("✅ Model trained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to train model")
        
        st.markdown("---")
        
        # Load model button
        if st.button("📂 Load Saved Model", use_container_width=True):
            with st.spinner("Loading model..."):
                if classifier.load_model():
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ No saved model found")
        
        # Save model button
        if classifier.trained:
            st.markdown("---")
            if st.button("💾 Save Model", use_container_width=True):
                if classifier.save_model():
                    st.success("✅ Model saved!")
                else:
                    st.warning("⚠️ Could not save model")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Performance", 
        "🔮 Predict New Institution",
        "📈 Visualizations"
    ])
    
    with tab1:
        st.header("Model Performance Metrics")
        
        if not classifier.trained:
            st.warning("⚠️ Please train or load a model first")
        else:
            # Display metrics
            st.subheader("Model Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{classifier.metrics['accuracy']:.2%}")
            
            with col2:
                st.metric("Training Samples", classifier.metrics['training_samples'])
            
            with col3:
                st.metric("Test Samples", classifier.metrics['test_samples'])
            
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
        
        if not classifier.trained:
            st.error("❌ Model not trained or loaded!")
            st.info("📋 Please go to the sidebar and:")
            st.info("1. Click '🚀 Train Model' to train a new model")
            st.info("2. OR Click '📂 Load Saved Model' to load an existing model")
            return
        
        st.success("✅ Model is ready for predictions!")
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Enter Institution Metrics")
            
            # Organize features into sections
            sections = {
                "Academic Metrics": [
                    ('student_faculty_ratio', 'slider', (5.0, 40.0, 15.0, 0.1)),
                    ('phd_faculty_ratio', 'slider', (0.1, 1.0, 0.5, 0.01)),
                    ('higher_education_rate', 'slider', (5.0, 50.0, 20.0, 0.1))
                ],
                "Research Metrics": [
                    ('research_publications', 'number', (0, 100, 20, 1)),
                    ('research_grants_amount', 'number', (0, 1000000, 100000, 10000)),
                    ('patents_filed', 'number', (0, 20, 3, 1)),
                    ('industry_collaborations', 'number', (0, 20, 5, 1))
                ],
                "Infrastructure Metrics": [
                    ('digital_infrastructure_score', 'slider', (1.0, 10.0, 6.0, 0.1)),
                    ('library_volumes', 'number', (1000, 50000, 10000, 1000)),
                    ('laboratory_equipment_score', 'slider', (1.0, 10.0, 7.0, 0.1))
                ],
                "Administrative Metrics": [
                    ('financial_stability_score', 'slider', (1.0, 10.0, 7.0, 0.1)),
                    ('compliance_score', 'slider', (1.0, 10.0, 7.0, 0.1)),
                    ('administrative_efficiency', 'slider', (1.0, 10.0, 6.5, 0.1))
                ],
                "Placement Metrics": [
                    ('placement_rate', 'slider', (40.0, 100.0, 75.0, 0.1)),
                    ('entrepreneurship_cell_score', 'slider', (1.0, 10.0, 6.0, 0.1))
                ],
                "Outreach Metrics": [
                    ('community_projects', 'number', (0, 20, 5, 1)),
                    ('rural_outreach_score', 'slider', (1.0, 10.0, 6.0, 0.1)),
                    ('inclusive_education_index', 'slider', (1.0, 10.0, 6.5, 0.1))
                ],
                "Overall Performance": [
                    ('performance_score', 'slider', (1.0, 10.0, 5.5, 0.1))
                ]
            }
            
            input_data = {}
            
            for section_name, features in sections.items():
                st.markdown(f"### {section_name}")
                cols = st.columns(min(len(features), 3))
                
                for idx, (feature_name, input_type, params) in enumerate(features):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        if input_type == 'slider':
                            min_val, max_val, default_val, step = params
                            value = st.slider(
                                feature_name.replace('_', ' ').title(),
                                min_val, max_val, default_val, step
                            )
                        else:  # number
                            min_val, max_val, default_val, step = params
                            value = st.number_input(
                                feature_name.replace('_', ' ').title(),
                                min_val, max_val, default_val, step
                            )
                        input_data[feature_name] = value
            
            # Predict button
            predict_button = st.form_submit_button("🔮 Predict Risk Level", 
                                                 type="primary", 
                                                 use_container_width=True)
            
            if predict_button:
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
                            <p style='font-size: 14px; color: #6c757d;'>Based on analysis of {len(classifier.features)} performance metrics</p>
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
    
    with tab3:
        st.header("Model Visualizations")
        
        if not classifier.trained:
            st.warning("⚠️ Please train or load a model first")
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
