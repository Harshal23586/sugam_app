# modules/decision_tree_classifier.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class InstitutionalDecisionTreeClassifier:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
        # Initialize model components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.features = None
        self.metrics = {}
        self.training_history = []
        
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
        if 'dt_training_history' in st.session_state:
            self.training_history = st.session_state.dt_training_history
        
        self.model_path = "models/decision_tree_model.pkl"
        
    @property
    def trained(self):
        """Check if model is trained"""
        return st.session_state.get('dt_model_trained', False)
    
    def prepare_data(self, feature_selection=None):
        """Prepare data for decision tree training with optional feature selection"""
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
            
            # Apply feature selection if specified
            if feature_selection and feature_selection != 'all':
                if feature_selection == 'academic':
                    academic_features = ['student_faculty_ratio', 'phd_faculty_ratio', 
                                       'higher_education_rate', 'performance_score']
                    selected_features = [f for f in available_features if f in academic_features]
                elif feature_selection == 'research':
                    research_features = ['research_publications', 'research_grants_amount',
                                       'patents_filed', 'industry_collaborations']
                    selected_features = [f for f in available_features if f in research_features]
                elif feature_selection == 'infrastructure':
                    infra_features = ['digital_infrastructure_score', 'library_volumes',
                                    'laboratory_equipment_score']
                    selected_features = [f for f in available_features if f in infra_features]
                elif feature_selection == 'administrative':
                    admin_features = ['financial_stability_score', 'compliance_score',
                                    'administrative_efficiency', 'placement_rate']
                    selected_features = [f for f in available_features if f in admin_features]
                elif feature_selection == 'top10':
                    # We'll determine top features after feature importance analysis
                    selected_features = available_features  # Placeholder
                else:
                    selected_features = available_features
            else:
                selected_features = available_features
            
            st.write(f"✅ Selected features: {len(selected_features)}")
            st.write(f"📋 Features list: {selected_features}")
            
            # Remove rows with missing values in target
            df = df.dropna(subset=['risk_level'])
            
            # Handle missing values in features
            df_clean = df.copy()
            missing_counts = {}
            for feature in selected_features:
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
            self.features = selected_features
            
            st.write(f"📊 Clean data shape: {df_clean.shape}")
            st.write(f"🎯 Classes: {self.label_encoder.classes_}")
            
            # Show class distribution
            class_dist = df_clean['risk_level'].value_counts()
            st.write(f"📈 Class distribution:")
            for cls, count in class_dist.items():
                percentage = (count / len(df_clean)) * 100
                st.write(f"   - {cls}: {count} ({percentage:.1f}%)")
            
            return df_clean[selected_features], df_clean['risk_level_encoded']
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def train_model(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                   max_features=None, criterion='gini', test_size=0.2, feature_selection=None,
                   cross_validation=False, cv_folds=5):
        """Train decision tree model with enhanced options"""
        try:
            with st.spinner("Preparing data..."):
                X, y = self.prepare_data(feature_selection)
            
            if X is None or y is None:
                st.error("Could not prepare data for training")
                return False
            
            if len(X) == 0:
                st.error("No data available for training")
                return False
            
            st.success(f"✅ Data prepared: {X.shape[0]} samples × {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            st.write(f"📚 Training samples: {X_train.shape[0]}")
            st.write(f"🧪 Test samples: {X_test.shape[0]}")
            
            # Display training parameters
            st.info(f"**Training Parameters:**")
            params_info = f"""
            - Max Depth: {max_depth if max_depth else 'None (unlimited)'}
            - Min Samples Split: {min_samples_split}
            - Min Samples Leaf: {min_samples_leaf}
            - Max Features: {max_features if max_features else 'All'}
            - Criterion: {criterion}
            - Test Size: {test_size}
            - Feature Selection: {feature_selection if feature_selection else 'All features'}
            """
            st.markdown(params_info)
            
            # Train model
            with st.spinner("Training decision tree model..."):
                self.model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X_train, y_train)
            
            # Cross-validation if requested
            cv_scores = None
            if cross_validation and cv_folds > 1:
                with st.spinner(f"Performing {cv_folds}-fold cross-validation..."):
                    cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Get detailed classification report
            report_dict = classification_report(y_test, y_pred_test, 
                                               target_names=self.label_encoder.classes_,
                                               output_dict=True)
            
            # Calculate additional metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred_test, average='weighted'
            )
            
            # Store metrics
            self.metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': report_dict,
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
                'classes': self.label_encoder.classes_.tolist(),
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_count': X.shape[1],
                'cv_scores': cv_scores,
                'tree_depth': self.model.get_depth(),
                'n_leaves': self.model.get_n_leaves(),
                'parameters': {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'criterion': criterion,
                    'test_size': test_size,
                    'feature_selection': feature_selection,
                    'cross_validation': cross_validation,
                    'cv_folds': cv_folds
                }
            }
            
            # Store in training history
            training_record = {
                'timestamp': pd.Timestamp.now(),
                'metrics': self.metrics.copy(),
                'parameters': self.metrics['parameters'].copy()
            }
            self.training_history.append(training_record)
            
            # Save to session state
            st.session_state.dt_model = self.model
            st.session_state.dt_label_encoder = self.label_encoder
            st.session_state.dt_features = self.features
            st.session_state.dt_metrics = self.metrics
            st.session_state.dt_training_history = self.training_history
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
                    'metrics': self.metrics,
                    'training_history': self.training_history
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
                self.training_history = saved_data.get('training_history', [])
                
                # Update session state
                st.session_state.dt_model = self.model
                st.session_state.dt_label_encoder = self.label_encoder
                st.session_state.dt_features = self.features
                st.session_state.dt_metrics = self.metrics
                st.session_state.dt_training_history = self.training_history
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
            'n_leaves': self.model.get_n_leaves() if self.model else None,
            'train_accuracy': self.metrics.get('train_accuracy', 0) if self.metrics else 0,
            'test_accuracy': self.metrics.get('test_accuracy', 0) if self.metrics else 0,
            'training_samples': self.metrics.get('training_samples', 0),
            'test_samples': self.metrics.get('test_samples', 0),
            'training_history_count': len(self.training_history)
        }
        
        return info

def create_decision_tree_module(analyzer):
    """Create Streamlit interface for decision tree classification"""
    
    st.title("🌳 Decision Tree Classifier - Risk Level Prediction")
    st.markdown("""
    This module trains a Decision Tree model to predict institutional risk levels based on 
    various performance metrics. The model learns patterns from historical data to predict 
    whether an institution falls into **Critical Risk**, **High Risk**, **Medium Risk**, or **Low Risk** categories.
    """)
    st.markdown("---")
    
    # Initialize classifier
    classifier = InstitutionalDecisionTreeClassifier(analyzer)
    
    # Display data info
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(analyzer.historical_data))
    with col2:
        st.metric("Unique Institutions", analyzer.historical_data['institution_id'].nunique())
    with col3:
        st.metric("Data Years", f"{analyzer.historical_data['year'].min()}-{analyzer.historical_data['year'].max()}")
    with col4:
        risk_counts = analyzer.historical_data['risk_level'].value_counts()
        st.metric("Risk Categories", len(risk_counts))
    
    # Display risk distribution
    st.subheader("📈 Risk Level Distribution")
    risk_dist = analyzer.historical_data['risk_level'].value_counts()
    fig_dist = px.pie(values=risk_dist.values, names=risk_dist.index, 
                     title="Distribution of Risk Levels in Dataset",
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Check if model is trained
    if classifier.trained:
        st.success("✅ Model is trained and ready!")
        model_info = classifier.get_model_info()
        if model_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{model_info['test_accuracy']:.1%}")
            with col2:
                st.metric("Features", model_info['feature_count'])
            with col3:
                st.metric("Tree Depth", model_info['tree_depth'])
            with col4:
                st.metric("Training Runs", model_info['training_history_count'])
    else:
        st.warning("⚠️ Model not trained yet. Please train the model in the '🚀 Train Model' tab.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Train Model", 
        "📊 Model Performance", 
        "🔮 Predict New Institution",
        "📈 Visualizations"
    ])
    
    with tab1:
        st.header("🚀 Train Decision Tree Model")
        st.markdown("""
        Configure the training parameters below and train the decision tree model. 
        The model will learn to predict risk levels based on historical institution data.
        """)
        
        # Training configuration in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌳 Tree Parameters")
            
            max_depth = st.slider("Max Tree Depth", 3, 30, 10, 
                                 help="Maximum depth of the decision tree. Deeper trees can capture more complex patterns but may overfit.")
            
            min_samples_split = st.slider("Min Samples Split", 2, 50, 5,
                                         help="Minimum number of samples required to split an internal node")
            
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 2,
                                        help="Minimum number of samples required to be at a leaf node")
            
            criterion = st.selectbox("Split Criterion", 
                                   ['gini', 'entropy'],
                                   help="Function to measure the quality of a split")
            
            max_features = st.selectbox("Max Features per Split",
                                      [None, 'sqrt', 'log2', 0.5, 0.7, 0.9],
                                      format_func=lambda x: 'All' if x is None else f'{x}',
                                      help="Number of features to consider when looking for the best split")
        
        with col2:
            st.subheader("📊 Training Parameters")
            
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05,
                                 help="Proportion of data to use for testing")
            
            feature_selection = st.selectbox("Feature Selection Strategy",
                                           ['all', 'academic', 'research', 'infrastructure', 'administrative'],
                                           help="Select which features to use for training")
            
            st.subheader("🔬 Advanced Options")
            
            cross_validation = st.checkbox("Enable Cross-Validation",
                                         help="Perform k-fold cross-validation during training")
            
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5,
                                disabled=not cross_validation,
                                help="Number of folds for cross-validation")
        
        # Display current data info
        st.subheader("📋 Data Preview")
        with st.expander("View Data Sample"):
            st.dataframe(analyzer.historical_data.head())
        
        # Show feature correlations
        st.subheader("🔗 Feature Correlations")
        if st.button("Show Feature Correlation Matrix"):
            numeric_cols = analyzer.historical_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = analyzer.historical_data[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, square=True, linewidths=0.5)
                ax.set_title('Feature Correlation Matrix')
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric features for correlation analysis")
        
        # Train button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button("🚀 Start Training", 
                                    type="primary", 
                                    use_container_width=True,
                                    help="Click to train the decision tree model with the configured parameters")
        
        if train_button:
            with st.spinner("Training decision tree model. This may take a moment..."):
                if classifier.train_model(max_depth=max_depth, 
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features,
                                         criterion=criterion,
                                         test_size=test_size,
                                         feature_selection=feature_selection,
                                         cross_validation=cross_validation,
                                         cv_folds=cv_folds):
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    
                    # Show quick results
                    model_info = classifier.get_model_info()
                    if model_info:
                        st.subheader("🎯 Quick Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Accuracy", f"{model_info['test_accuracy']:.1%}")
                        with col2:
                            st.metric("Tree Depth", model_info['tree_depth'])
                        with col3:
                            st.metric("Features Used", model_info['feature_count'])
                    
                    # Auto-switch to performance tab
                    st.info("📊 Switch to the 'Model Performance' tab to see detailed metrics.")
                else:
                    st.error("❌ Failed to train model. Please check the data and parameters.")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        if not classifier.trained:
            st.warning("⚠️ Please train or load a model first")
            st.info("📋 Go to the '🚀 Train Model' tab to train a new model")
        else:
            # Display metrics in cards
            st.subheader("🎯 Model Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test Accuracy", f"{classifier.metrics['test_accuracy']:.2%}")
            
            with col2:
                st.metric("Training Accuracy", f"{classifier.metrics['train_accuracy']:.2%}")
            
            with col3:
                st.metric("F1 Score", f"{classifier.metrics['f1_score']:.2%}")
            
            with col4:
                st.metric("Precision", f"{classifier.metrics['precision']:.2%}")
            
            # Show overfitting indicator
            accuracy_diff = classifier.metrics['train_accuracy'] - classifier.metrics['test_accuracy']
            if accuracy_diff > 0.15:
                st.error(f"⚠️ Potential Overfitting: Training accuracy is {accuracy_diff:.1%} higher than test accuracy")
            elif accuracy_diff > 0.05:
                st.warning(f"⚠️ Moderate Overfitting: Training accuracy is {accuracy_diff:.1%} higher than test accuracy")
            else:
                st.success(f"✅ Good Generalization: Training and test accuracy are close")
            
            # Cross-validation results if available
            if classifier.metrics.get('cv_scores') is not None:
                st.subheader("🔬 Cross-Validation Results")
                cv_scores = classifier.metrics['cv_scores']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean CV Accuracy", f"{cv_scores.mean():.2%}")
                with col2:
                    st.metric("Std CV Accuracy", f"{cv_scores.std():.2%}")
                with col3:
                    st.metric("CV Fold Count", len(cv_scores))
                
                # Plot CV scores
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(1, len(cv_scores) + 1), cv_scores)
                ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.2%}')
                ax.set_xlabel('Fold')
                ax.set_ylabel('Accuracy')
                ax.set_title('Cross-Validation Scores Across Folds')
                ax.legend()
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            
            # Classification report
            st.subheader("📋 Classification Report")
            if 'classification_report' in classifier.metrics:
                report_df = pd.DataFrame(classifier.metrics['classification_report']).transpose()
                # Remove support column for better display
                if 'support' in report_df.columns:
                    support_series = report_df['support']
                    report_df = report_df.drop(columns=['support'])
                
                st.dataframe(report_df.style.format({
                    'precision': '{:.1%}',
                    'recall': '{:.1%}',
                    'f1-score': '{:.1%}'
                }).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))
                
                # Show support separately
                if 'support' in locals():
                    st.write("**Support (sample counts):**")
                    st.write(support_series.astype(int))
            
            # Confusion matrix
            st.subheader("🎯 Confusion Matrix")
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
            st.subheader("📊 Feature Importance Analysis")
            importance_df = classifier.get_feature_importance()
            if importance_df is not None:
                # Bar chart
                fig = px.bar(importance_df.head(15), 
                            x='importance', y='feature',
                            orientation='h',
                            title="Top 15 Most Important Features",
                            color='importance',
                            color_continuous_scale='viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cumulative importance
                importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
                fig_cum = px.line(importance_df, x=range(1, len(importance_df) + 1), 
                                 y='cumulative_importance',
                                 title="Cumulative Feature Importance",
                                 labels={'x': 'Number of Features', 'y': 'Cumulative Importance'})
                fig_cum.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                annotation_text="80% threshold")
                fig_cum.update_layout(height=300)
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Show full table
                with st.expander("📋 View All Feature Importances"):
                    st.dataframe(importance_df.style.format({'importance': '{:.4f}', 
                                                           'cumulative_importance': '{:.1%}'}))
            
            # Training history
            if classifier.training_history:
                st.subheader("📚 Training History")
                history_df = pd.DataFrame([
                    {
                        'Run': i+1,
                        'Timestamp': rec['timestamp'],
                        'Test Accuracy': rec['metrics']['test_accuracy'],
                        'Train Accuracy': rec['metrics']['train_accuracy'],
                        'Features': rec['metrics']['feature_count'],
                        'Max Depth': rec['parameters']['max_depth'],
                        'Criterion': rec['parameters']['criterion']
                    }
                    for i, rec in enumerate(classifier.training_history)
                ])
                
                if not history_df.empty:
                    st.dataframe(history_df.style.format({
                        'Test Accuracy': '{:.2%}',
                        'Train Accuracy': '{:.2%}'
                    }))
                    
                    # Plot accuracy history
                    fig_history = px.line(history_df, x='Run', y=['Test Accuracy', 'Train Accuracy'],
                                         title="Accuracy Over Training Runs",
                                         markers=True)
                    fig_history.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_history, use_container_width=True)
    
    with tab3:
        st.header("Predict Risk for New Institution")
        
        if not classifier.trained:
            st.error("❌ Model not trained or loaded!")
            st.info("📋 Please:")
            st.info("1. Go to the '🚀 Train Model' tab and train a new model")
            st.info("2. OR Use the sidebar to load a saved model")
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
                            risk_desc = "Immediate action required"
                        elif 'High' in risk_level:
                            color = '#fd7e14'
                            icon = '🟠'
                            emoji = '⚠️'
                            risk_desc = "Close monitoring needed"
                        elif 'Medium' in risk_level:
                            color = '#ffc107'
                            icon = '🟡'
                            emoji = '📊'
                            risk_desc = "Regular monitoring recommended"
                        else:
                            color = '#28a745'
                            icon = '🟢'
                            emoji = '✅'
                            risk_desc = "Good performance"
                        
                        # Main result card
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {color};'>
                            <h1 style='color: {color}; margin-bottom: 10px;'>{emoji} {risk_level}</h1>
                            <p style='font-size: 18px; margin-bottom: 5px;'><strong>Confidence:</strong> {confidence:.1%}</p>
                            <p style='font-size: 16px; margin-bottom: 5px;'><strong>Assessment:</strong> {risk_desc}</p>
                            <p style='font-size: 14px; color: #6c757d;'>Based on analysis of {len(classifier.features)} performance metrics</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Quick recommendations
                            recommendations = {
                                'Critical Risk': "Immediate improvements required. Consider external audit and intervention.",
                                'High Risk': "Submit improvement plan within 30 days. Monthly monitoring required.",
                                'Medium Risk': "Quarterly review recommended. Address specific weaknesses.",
                                'Low Risk': "Annual review sufficient. Continue best practices."
                            }
                            
                            recommendation = recommendations.get(risk_level, "Review all metrics for improvement.")
                            st.info(f"**Recommendation:** {recommendation}")
                        
                        # Probability distribution
                        st.subheader("📈 Risk Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Risk Level': list(prediction['probabilities'].keys()),
                            'Probability': list(prediction['probabilities'].values())
                        }).sort_values('Probability', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(prob_df, 
                                    x='Risk Level', 
                                    y='Probability',
                                    color='Probability',
                                    color_continuous_scale='RdYlGn_r',
                                    title="Probability Distribution Across Risk Levels")
                        fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed probabilities
                        with st.expander("📋 View Detailed Probabilities"):
                            for risk_class, prob in prediction['probabilities'].items():
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.write(f"**{risk_class}**")
                                with col2:
                                    st.write(f"{prob:.1%}")
                        
                        # Feature impact analysis
                        st.subheader("🔍 Key Contributing Factors")
                        if classifier.model is not None:
                            # Get feature importance for this specific prediction
                            feature_values = pd.DataFrame([input_data])
                            importance_df = classifier.get_feature_importance()
                            
                            if importance_df is not None:
                                # Show top 5 features that contributed most
                                top_features = importance_df.head(5)
                                st.write("Top features influencing this prediction:")
                                
                                for _, row in top_features.iterrows():
                                    feature_name = row['feature']
                                    importance = row['importance']
                                    value = input_data.get(feature_name, 'N/A')
                                    
                                    col1, col2, col3 = st.columns([3, 2, 2])
                                    with col1:
                                        st.write(f"**{feature_name.replace('_', ' ').title()}**")
                                    with col2:
                                        st.write(f"Value: {value}")
                                    with col3:
                                        st.write(f"Impact: {importance:.1%}")
    
    with tab4:
        st.header("Model Visualizations")
        
        if not classifier.trained:
            st.warning("⚠️ Please train or load a model first")
        else:
            # Decision tree visualization
            st.subheader("🌳 Decision Tree Structure")
            viz_depth = st.slider("Visualization Depth", 2, 5, 3,
                                 help="Shallow trees are easier to visualize",
                                 key="viz_depth")
            
            fig = classifier.visualize_tree(max_depth=viz_depth)
            if fig:
                st.pyplot(fig)
                st.caption("Note: Full tree visualization might be complex. Consider using depth 3-4 for clarity.")
            
            # Additional visualizations
            st.subheader("📊 Additional Visualizations")
            
            # Feature importance radial chart
            importance_df = classifier.get_feature_importance()
            if importance_df is not None and len(importance_df) > 0:
                # Top 10 features for radial chart
                top_features = importance_df.head(10)
                
                fig_radial = go.Figure()
                fig_radial.add_trace(go.Scatterpolar(
                    r=top_features['importance'].values,
                    theta=top_features['feature'].str.replace('_', ' ').str.title(),
                    fill='toself',
                    name='Feature Importance'
                ))
                
                fig_radial.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, top_features['importance'].max() * 1.1]
                        )),
                    showlegend=False,
                    title="Top 10 Features - Radial View"
                )
                
                st.plotly_chart(fig_radial, use_container_width=True)
            
            # Tree depth vs accuracy analysis
            st.subheader("🔬 Tree Complexity Analysis")
            
            if classifier.training_history and len(classifier.training_history) > 1:
                # Extract depth and accuracy from history
                depths = []
                accuracies = []
                
                for record in classifier.training_history:
                    if 'tree_depth' in record['metrics'] and 'test_accuracy' in record['metrics']:
                        depths.append(record['metrics']['tree_depth'])
                        accuracies.append(record['metrics']['test_accuracy'])
                
                if depths and accuracies:
                    analysis_df = pd.DataFrame({
                        'Tree Depth': depths,
                        'Test Accuracy': accuracies
                    })
                    
                    fig_complexity = px.scatter(analysis_df, x='Tree Depth', y='Test Accuracy',
                                               title="Tree Depth vs Accuracy",
                                               trendline="lowess")
                    fig_complexity.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_complexity, use_container_width=True)
            
            # Model comparison if multiple runs
            if classifier.training_history and len(classifier.training_history) > 1:
                st.subheader("📈 Model Performance Comparison")
                
                comparison_data = []
                for i, record in enumerate(classifier.training_history[-5:]):  # Last 5 runs
                    comparison_data.append({
                        'Run': f"Run {i+1}",
                        'Test Accuracy': record['metrics']['test_accuracy'],
                        'Train Accuracy': record['metrics']['train_accuracy'],
                        'Depth': record['metrics']['tree_depth'],
                        'Features': record['metrics']['feature_count']
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig_comparison = px.bar(comparison_df, 
                                           x='Run', 
                                           y=['Test Accuracy', 'Train Accuracy'],
                                           barmode='group',
                                           title="Accuracy Comparison Across Training Runs")
                    fig_comparison.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🛠️ Model Controls")
        
        # Quick training
        st.subheader("Quick Training")
        
        quick_max_depth = st.slider("Quick Max Depth", 3, 15, 8, 
                                   help="Maximum depth for quick training",
                                   key="quick_depth")
        
        if st.button("⚡ Quick Train", use_container_width=True):
            with st.spinner("Quick training in progress..."):
                if classifier.train_model(max_depth=quick_max_depth):
                    st.success("✅ Model trained!")
                    st.rerun()
                else:
                    st.error("❌ Training failed")
        
        st.markdown("---")
        
        # Model management
        st.subheader("Model Management")
        
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
            if st.button("💾 Save Model", use_container_width=True):
                if classifier.save_model():
                    st.success("✅ Model saved!")
                else:
                    st.warning("⚠️ Could not save model")
        
        # Reset model
        if st.button("🔄 Reset Model", use_container_width=True):
            st.session_state.dt_model_trained = False
            st.session_state.dt_model = None
            st.session_state.dt_metrics = {}
            st.rerun()
        
        st.markdown("---")
        
        # Model info
        st.subheader("Model Information")
        if classifier.trained:
            model_info = classifier.get_model_info()
            if model_info:
                st.write(f"**Type:** {model_info['model_type']}")
                st.write(f"**Features:** {model_info['feature_count']}")
                st.write(f"**Tree Depth:** {model_info['tree_depth']}")
                st.write(f"**Leaves:** {model_info.get('n_leaves', 'N/A')}")
                st.write(f"**Accuracy:** {model_info['test_accuracy']:.1%}")
                st.write(f"**Training Runs:** {model_info['training_history_count']}")
