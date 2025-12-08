# rag_score (2).py
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DocumentFormValidator:
    """RAG-based validator to compare document-extracted data with form-submitted data"""
    
    def __init__(self):
        self.embedding_model = None
        self.initialize_embeddings()
        self.thresholds = {
            'high_similarity': 0.85,
            'medium_similarity': 0.70,
            'low_similarity': 0.50
        }
    
    def initialize_embeddings(self):
        """Initialize embedding model with fallback"""
        try:
            # Use lightweight model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.sidebar.success("✅ RAG Validator Initialized")
        except:
            st.sidebar.warning("⚠️ Using basic text matching for validation")
            self.embedding_model = None
    
    def extract_document_data(self, uploaded_files: List) -> Dict[str, Any]:
        """Extract structured data from uploaded documents"""
        document_data = {
            'academic_metrics': {},
            'research_metrics': {},
            'infrastructure_metrics': {},
            'governance_metrics': {},
            'financial_metrics': {},
            'student_metrics': {},
            'extracted_text': ""
        }
        
        all_text = ""
        for file in uploaded_files:
            try:
                # Extract text based on file type
                text = self.extract_text_from_file(file)
                if text:
                    all_text += text + "\n\n"
                    
                    # Extract structured data using patterns
                    self.extract_from_text(text, document_data)
                    
            except Exception as e:
                st.warning(f"Could not process {file.name}: {str(e)}")
                continue
        
        document_data['extracted_text'] = all_text
        
        # Clean and normalize extracted data
        document_data = self.clean_extracted_data(document_data)
        
        return document_data
    
    def extract_text_from_file(self, file) -> str:
        """Extract text from different file types"""
        text = ""
        
        if file.name.lower().endswith('.pdf'):
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except:
                # Fallback: return filename as text
                text = f"Document: {file.name}\nType: PDF\n"
        
        elif file.name.lower().endswith(('.doc', '.docx')):
            try:
                import docx
                doc = docx.Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            except:
                text = f"Document: {file.name}\nType: Word Document\n"
        
        elif file.name.lower().endswith(('.txt', '.csv')):
            text = file.getvalue().decode('utf-8', errors='ignore')
        
        elif file.name.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file)
                text = df.to_string()
            except:
                text = f"Document: {file.name}\nType: Excel File\n"
        
        else:
            text = f"Document: {file.name}\nType: {file.type}\n"
        
        return text
    
    def extract_from_text(self, text: str, document_data: Dict):
        """Extract structured data from text using pattern matching"""
        
        # Academic metrics patterns
        academic_patterns = {
            'naac_grade': r'NAAC\s*(?:grade|accreditation|score)[:\s]*([A+]+)',
            'nirf_ranking': r'NIRF\s*(?:rank|ranking)[:\s]*(\d+)',
            'student_faculty_ratio': r'(?:student|student-faculty)\s*(?:ratio|ratio:)[:\s]*(\d+(?:\.\d+)?)',
            'phd_faculty_ratio': r'PhD\s*(?:faculty|ratio)[:\s]*(\d+(?:\.\d+)?)%',
            'placement_rate': r'placement\s*(?:rate|percentage)[:\s]*(\d+(?:\.\d+)?)%'
        }
        
        for key, pattern in academic_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                document_data['academic_metrics'][key] = matches[0]
        
        # Research metrics patterns
        research_patterns = {
            'research_publications': r'research\s*(?:publications|papers)[:\s]*(\d+)',
            'research_grants': r'research\s*(?:grants|funding)[:\s]*[₹$]?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            'patents_filed': r'patents?\s*(?:filed|granted)[:\s]*(\d+)'
        }
        
        for key, pattern in research_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                document_data['research_metrics'][key] = matches[0]
        
        # Infrastructure metrics
        infra_patterns = {
            'digital_infrastructure_score': r'digital\s*(?:infrastructure|infra)[\s:-]*(\d+(?:\.\d+)?)/10',
            'library_volumes': r'library\s*(?:volumes|books)[:\s]*(\d+(?:,\d+)*)',
            'campus_area': r'campus\s*(?:area|size)[:\s]*(\d+(?:\.\d+)?)\s*(?:acres|hectares)'
        }
        
        for key, pattern in infra_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                document_data['infrastructure_metrics'][key] = matches[0]
        
        # Financial metrics
        financial_patterns = {
            'financial_stability_score': r'financial\s*(?:stability|health)[\s:-]*(\d+(?:\.\d+)?)/10',
            'annual_budget': r'(?:annual\s*)?budget[:\s]*[₹$]?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            'research_investment': r'research\s*(?:investment|spending)[:\s]*[₹$]?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        }
        
        for key, pattern in financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                document_data['financial_metrics'][key] = matches[0]
    
    def clean_extracted_data(self, document_data: Dict) -> Dict:
        """Clean and normalize extracted data"""
        cleaned_data = {}
        
        for category, data in document_data.items():
            if category == 'extracted_text':
                cleaned_data[category] = data
                continue
            
            cleaned_data[category] = {}
            for key, value in data.items():
                # Convert percentages to decimal
                if key.endswith('_rate') or key.endswith('_ratio'):
                    if '%' in str(value):
                        value = value.replace('%', '')
                    try:
                        value = float(value) / 100 if float(value) > 1 else float(value)
                    except:
                        pass
                
                # Remove currency symbols and commas
                if any(curr in str(value) for curr in ['₹', '$', '€', '£']):
                    value = re.sub(r'[₹$€£,]', '', str(value))
                
                cleaned_data[category][key] = value
        
        return cleaned_data
    
    def compare_with_form_data(self, document_data: Dict, form_data: Dict) -> Dict[str, Any]:
        """Compare document-extracted data with form-submitted data"""
        
        comparison_results = {
            'matches': [],
            'mismatches': [],
            'missing_in_document': [],
            'missing_in_form': [],
            'similarity_scores': {},
            'overall_match_percentage': 0
        }
        
        total_metrics = 0
        matched_metrics = 0
        
        # Compare by category
        categories = ['academic_metrics', 'research_metrics', 'infrastructure_metrics', 
                     'financial_metrics', 'governance_metrics', 'student_metrics']
        
        for category in categories:
            doc_metrics = document_data.get(category, {})
            form_metrics = form_data.get(category, {})
            
            # All unique metrics across both sources
            all_metrics = set(list(doc_metrics.keys()) + list(form_metrics.keys()))
            
            for metric in all_metrics:
                total_metrics += 1
                doc_value = doc_metrics.get(metric)
                form_value = form_metrics.get(metric)
                
                if doc_value is None and form_value is not None:
                    comparison_results['missing_in_document'].append({
                        'category': category,
                        'metric': metric,
                        'form_value': form_value
                    })
                elif form_value is None and doc_value is not None:
                    comparison_results['missing_in_form'].append({
                        'category': category,
                        'metric': metric,
                        'document_value': doc_value
                    })
                elif doc_value is not None and form_value is not None:
                    # Calculate similarity
                    similarity = self.calculate_similarity(doc_value, form_value)
                    comparison_results['similarity_scores'][f"{category}.{metric}"] = similarity
                    
                    if similarity >= self.thresholds['high_similarity']:
                        comparison_results['matches'].append({
                            'category': category,
                            'metric': metric,
                            'document_value': doc_value,
                            'form_value': form_value,
                            'similarity': similarity
                        })
                        matched_metrics += 1
                    else:
                        comparison_results['mismatches'].append({
                            'category': category,
                            'metric': metric,
                            'document_value': doc_value,
                            'form_value': form_value,
                            'similarity': similarity,
                            'discrepancy': self.calculate_discrepancy(doc_value, form_value)
                        })
        
        # Calculate overall match percentage
        if total_metrics > 0:
            comparison_results['overall_match_percentage'] = (matched_metrics / total_metrics) * 100
        
        # Calculate confidence score
        comparison_results['confidence_score'] = self.calculate_confidence_score(comparison_results)
        
        return comparison_results
    
    def calculate_similarity(self, value1, value2) -> float:
        """Calculate similarity between two values"""
        try:
            # Convert to strings for comparison
            str1 = str(value1).lower().strip()
            str2 = str(value2).lower().strip()
            
            # Exact match
            if str1 == str2:
                return 1.0
            
            # Numeric comparison
            try:
                num1 = float(str1)
                num2 = float(str2)
                
                # For percentages or ratios
                if abs(num1 - num2) <= 0.01:  # Very close
                    return 0.95
                elif abs(num1 - num2) <= 0.05:  # Close
                    return 0.85
                elif abs(num1 - num2) <= 0.10:  # Somewhat close
                    return 0.70
                else:
                    # Use relative difference
                    diff = abs(num1 - num2) / max(abs(num1), abs(num2))
                    return max(0, 1 - diff)
                    
            except ValueError:
                # Text similarity using embeddings if available
                if self.embedding_model:
                    embeddings = self.embedding_model.encode([str1, str2])
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    return float(similarity)
                else:
                    # Basic text similarity
                    words1 = set(str1.split())
                    words2 = set(str2.split())
                    if not words1 or not words2:
                        return 0.0
                    
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    return len(intersection) / len(union)
                    
        except Exception as e:
            return 0.0
    
    def calculate_discrepancy(self, value1, value2) -> str:
        """Calculate discrepancy between two values"""
        try:
            # Try numeric comparison
            num1 = float(value1)
            num2 = float(value2)
            
            diff = abs(num1 - num2)
            percentage_diff = (diff / max(abs(num1), abs(num2))) * 100
            
            if percentage_diff < 5:
                return "Minor difference"
            elif percentage_diff < 15:
                return "Moderate difference"
            elif percentage_diff < 30:
                return "Significant difference"
            else:
                return "Major discrepancy"
                
        except:
            # Text comparison
            if str(value1).lower() == str(value2).lower():
                return "Same value"
            else:
                return "Different values"
    
    def calculate_confidence_score(self, comparison_results: Dict) -> float:
        """Calculate overall confidence score for validation"""
        
        if not comparison_results['similarity_scores']:
            return 0.0
        
        # Get all similarity scores
        similarities = list(comparison_results['similarity_scores'].values())
        
        # Weighted average giving more weight to high similarities
        weighted_sum = 0
        weight_sum = 0
        
        for similarity in similarities:
            weight = similarity  # Higher similarity gets more weight
            weighted_sum += similarity * weight
            weight_sum += weight
        
        avg_similarity = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Penalize for missing data
        missing_penalty = len(comparison_results['missing_in_document']) * 0.05
        missing_penalty += len(comparison_results['missing_in_form']) * 0.03
        
        # Penalize for mismatches
        mismatch_penalty = len(comparison_results['mismatches']) * 0.02
        
        confidence = max(0, min(1, avg_similarity - missing_penalty - mismatch_penalty))
        
        return confidence
    
    def generate_validation_report(self, comparison_results: Dict, institution_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'institution_name': institution_name,
            'validation_date': datetime.now().isoformat(),
            'summary': {
                'total_metrics_compared': len(comparison_results['similarity_scores']),
                'matches': len(comparison_results['matches']),
                'mismatches': len(comparison_results['mismatches']),
                'missing_in_document': len(comparison_results['missing_in_document']),
                'missing_in_form': len(comparison_results['missing_in_form']),
                'overall_match_percentage': comparison_results['overall_match_percentage'],
                'confidence_score': comparison_results['confidence_score']
            },
            'category_breakdown': {},
            'recommendations': [],
            'risk_assessment': {}
        }
        
        # Categorize by match quality
        categories = {}
        for match in comparison_results['matches']:
            cat = match['category']
            if cat not in categories:
                categories[cat] = {'matches': 0, 'mismatches': 0, 'missing': 0}
            categories[cat]['matches'] += 1
        
        for mismatch in comparison_results['mismatches']:
            cat = mismatch['category']
            if cat not in categories:
                categories[cat] = {'matches': 0, 'mismatches': 0, 'missing': 0}
            categories[cat]['mismatches'] += 1
        
        for missing in comparison_results['missing_in_document']:
            cat = missing['category']
            if cat not in categories:
                categories[cat] = {'matches': 0, 'mismatches': 0, 'missing': 0}
            categories[cat]['missing'] += 1
        
        report['category_breakdown'] = categories
        
        # Generate recommendations
        confidence = comparison_results['confidence_score']
        
        if confidence >= 0.9:
            report['recommendations'].append("✅ High confidence validation - Form data strongly matches documents")
            report['recommendations'].append("Proceed with approval process")
        elif confidence >= 0.7:
            report['recommendations'].append("🟡 Moderate confidence validation - Some discrepancies found")
            report['recommendations'].append("Review mismatches before proceeding")
        elif confidence >= 0.5:
            report['recommendations'].append("🟠 Low confidence validation - Multiple discrepancies")
            report['recommendations'].append("Requires detailed verification")
        else:
            report['recommendations'].append("🔴 Poor validation - Significant discrepancies")
            report['recommendations'].append("Request resubmission or additional verification")
        
        # Generate risk assessment
        report['risk_assessment'] = {
            'validation_risk': 'Low' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'High',
            'data_consistency': 'High' if confidence >= 0.8 else 'Moderate' if confidence >= 0.6 else 'Low',
            'verification_required': len(comparison_results['mismatches']) > 3,
            'critical_issues': len([m for m in comparison_results['mismatches'] if m['similarity'] < 0.5])
        }
        
        return report

def create_rag_validation_dashboard(analyzer):
    """Create RAG-based validation dashboard"""
    
    st.header("🤖 RAG-Powered Document vs Form Data Validation")
    
    st.info("""
    **Upload institutional documents and compare with submitted form data**
    This system uses AI to extract data from documents and validate against form submissions.
    """)
    
    # Step 1: Select Institution
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
    selected_institution = st.selectbox(
        "Select Institution",
        current_institutions,
        key="validation_institution"
    )
    
    # Get institution name
    institution_data = analyzer.historical_data[
        (analyzer.historical_data['institution_id'] == selected_institution) &
        (analyzer.historical_data['year'] == 2023)
    ]
    
    institution_name = institution_data['institution_name'].iloc[0] if not institution_data.empty else "Unknown Institution"
    
    st.subheader(f"🏛️ Validating: {institution_name}")
    
    # Step 2: Document Upload
    st.subheader("📤 Step 1: Upload Institutional Documents")
    
    uploaded_files = st.file_uploader(
        "Upload Institutional Documents (PDF, DOC, TXT, Excel)",
        type=['pdf', 'doc', 'docx', 'txt', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload NAAC reports, annual reports, financial statements, etc."
    )
    
    # Step 3: Get Form Data
    st.subheader("📝 Step 2: Form Data for Validation")
    
    form_data = None
    form_options = ["Use sample form data", "Enter form data manually", "Load from database"]
    form_option = st.radio("Select form data source:", form_options, horizontal=True)
    
    if form_option == "Use sample form data":
        form_data = get_sample_form_data()
        st.json(form_data)
        
    elif form_option == "Enter form data manually":
        form_data = get_manual_form_input()
        
    elif form_option == "Load from database":
        # Try to get form submissions from database
        try:
            submissions = analyzer.get_institution_submissions(selected_institution)
            if not submissions.empty:
                latest_submission = json.loads(submissions.iloc[0]['submission_data'])
                form_data = latest_submission
                st.success("✅ Loaded latest submission from database")
                with st.expander("View loaded data"):
                    st.json(form_data)
            else:
                st.warning("No submissions found in database")
                form_data = get_sample_form_data()
        except:
            st.warning("Could not load from database")
            form_data = get_sample_form_data()
    
    # Step 4: Initialize Validator
    if 'validator' not in st.session_state:
        st.session_state.validator = DocumentFormValidator()
    
    validator = st.session_state.validator
    
    # Step 5: Run Validation
    st.subheader("🔍 Step 3: Run Validation")
    
    if st.button("🚀 Run Document vs Form Validation", type="primary"):
        if not uploaded_files:
            st.error("Please upload documents first")
            return
        
        if not form_data:
            st.error("Please provide form data")
            return
        
        with st.spinner("🤖 Extracting data from documents..."):
            # Extract data from documents
            document_data = validator.extract_document_data(uploaded_files)
            
            # Compare with form data
            comparison_results = validator.compare_with_form_data(document_data, form_data)
            
            # Generate report
            validation_report = validator.generate_validation_report(comparison_results, institution_name)
            
            # Store in session state
            st.session_state.validation_results = comparison_results
            st.session_state.validation_report = validation_report
            st.session_state.document_data = document_data
    
    # Display Results if available
    if 'validation_results' in st.session_state:
        display_validation_results(st.session_state)
    
    # Additional Analysis Tools
    st.markdown("---")
    st.subheader("📊 Additional Analysis Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Generate Detailed Report", type="secondary"):
            if 'validation_report' in st.session_state:
                display_detailed_report(st.session_state.validation_report)
    
    with col2:
        if st.button("⚠️ Show Discrepancies", type="secondary"):
            if 'validation_results' in st.session_state:
                display_discrepancies(st.session_state.validation_results)
    
    with col3:
        if st.button("💾 Export Validation Results", type="secondary"):
            if 'validation_report' in st.session_state:
                export_validation_results(st.session_state.validation_report)

def get_sample_form_data() -> Dict:
    """Get sample form data for testing"""
    return {
        'academic_metrics': {
            'naac_grade': 'A',
            'nirf_ranking': 45,
            'student_faculty_ratio': 18.5,
            'phd_faculty_ratio': 0.65,
            'placement_rate': 0.82
        },
        'research_metrics': {
            'research_publications': 125,
            'research_grants': 2500000,
            'patents_filed': 8
        },
        'infrastructure_metrics': {
            'digital_infrastructure_score': 8.5,
            'library_volumes': 25000,
            'campus_area': 65.5
        },
        'financial_metrics': {
            'financial_stability_score': 8.2,
            'annual_budget': 15000000,
            'research_investment': 2250000
        }
    }

def get_manual_form_input() -> Dict:
    """Get form data from manual input"""
    with st.form("manual_form_input"):
        st.write("### Academic Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            naac_grade = st.selectbox("NAAC Grade", ["A++", "A+", "A", "B++", "B+", "B", "C"])
            nirf_ranking = st.number_input("NIRF Ranking (if applicable)", min_value=1, max_value=200, value=None)
            student_faculty_ratio = st.number_input("Student-Faculty Ratio", min_value=5.0, max_value=50.0, value=20.0)
        
        with col2:
            phd_faculty_ratio = st.slider("PhD Faculty Ratio", 0.0, 1.0, 0.65, 0.01)
            placement_rate = st.slider("Placement Rate", 0.0, 1.0, 0.75, 0.01)
        
        st.write("### Research Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_publications = st.number_input("Research Publications", min_value=0, value=50)
        
        with col2:
            research_grants = st.number_input("Research Grants (₹)", min_value=0, value=1000000)
        
        with col3:
            patents_filed = st.number_input("Patents Filed", min_value=0, value=5)
        
        st.write("### Infrastructure Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            digital_infrastructure_score = st.slider("Digital Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
        
        with col2:
            library_volumes = st.number_input("Library Volumes", min_value=0, value=20000)
        
        with col3:
            campus_area = st.number_input("Campus Area (acres)", min_value=0.0, value=50.0)
        
        st.write("### Financial Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            financial_stability_score = st.slider("Financial Stability Score", 1.0, 10.0, 7.5, 0.1)
        
        with col2:
            annual_budget = st.number_input("Annual Budget (₹)", min_value=0, value=10000000)
        
        with col3:
            research_investment = st.number_input("Research Investment (₹)", min_value=0, value=1500000)
        
        submitted = st.form_submit_button("Submit Form Data")
        
        if submitted:
            return {
                'academic_metrics': {
                    'naac_grade': naac_grade,
                    'nirf_ranking': nirf_ranking if nirf_ranking else None,
                    'student_faculty_ratio': student_faculty_ratio,
                    'phd_faculty_ratio': phd_faculty_ratio,
                    'placement_rate': placement_rate
                },
                'research_metrics': {
                    'research_publications': research_publications,
                    'research_grants': research_grants,
                    'patents_filed': patents_filed
                },
                'infrastructure_metrics': {
                    'digital_infrastructure_score': digital_infrastructure_score,
                    'library_volumes': library_volumes,
                    'campus_area': campus_area
                },
                'financial_metrics': {
                    'financial_stability_score': financial_stability_score,
                    'annual_budget': annual_budget,
                    'research_investment': research_investment
                }
            }
    
    return None

def display_validation_results(session_state):
    """Display validation results"""
    
    st.markdown("---")
    st.subheader("📊 Validation Results")
    
    results = session_state.validation_results
    report = session_state.validation_report
    document_data = session_state.document_data
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence Score", f"{report['summary']['confidence_score']:.2%}")
    
    with col2:
        st.metric("Match Percentage", f"{report['summary']['overall_match_percentage']:.1f}%")
    
    with col3:
        st.metric("Matches", report['summary']['matches'])
    
    with col4:
        st.metric("Mismatches", report['summary']['mismatches'])
    
    # Confidence indicator
    confidence = report['summary']['confidence_score']
    if confidence >= 0.9:
        st.success("✅ **HIGH CONFIDENCE**: Form data strongly matches documents")
    elif confidence >= 0.7:
        st.info("🟡 **MODERATE CONFIDENCE**: Form data mostly matches documents")
    elif confidence >= 0.5:
        st.warning("🟠 **LOW CONFIDENCE**: Significant discrepancies found")
    else:
        st.error("🔴 **POOR CONFIDENCE**: Major discrepancies - verification required")
    
    # Category Breakdown
    st.subheader("📈 Category-wise Analysis")
    
    categories_data = []
    for category, counts in report['category_breakdown'].items():
        total = counts['matches'] + counts['mismatches'] + counts['missing']
        match_percentage = (counts['matches'] / total * 100) if total > 0 else 0
        
        categories_data.append({
            'Category': category.replace('_', ' ').title(),
            'Matches': counts['matches'],
            'Mismatches': counts['mismatches'],
            'Missing': counts['missing'],
            'Match %': f"{match_percentage:.1f}%"
        })
    
    categories_df = pd.DataFrame(categories_data)
    st.dataframe(categories_df, use_container_width=True)
    
    # Visual comparison
    with st.expander("📊 Visual Comparison"):
        # Create comparison chart
        comparison_data = []
        
        for match in results['matches'][:10]:  # Show top 10 matches
            comparison_data.append({
                'Metric': match['metric'],
                'Document Value': match['document_value'],
                'Form Value': match['form_value'],
                'Similarity': match['similarity'],
                'Status': '✅ Match'
            })
        
        for mismatch in results['mismatches'][:10]:  # Show top 10 mismatches
            comparison_data.append({
                'Metric': mismatch['metric'],
                'Document Value': mismatch['document_value'],
                'Form Value': mismatch['form_value'],
                'Similarity': mismatch['similarity'],
                'Status': '⚠️ Mismatch'
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Extracted Document Data
    with st.expander("📄 Extracted Document Data"):
        for category, data in document_data.items():
            if category == 'extracted_text':
                continue
            
            if data:
                st.write(f"**{category.replace('_', ' ').title()}:**")
                for key, value in data.items():
                    st.write(f"- {key}: {value}")
    
    # Recommendations
    with st.expander("💡 Recommendations"):
        for rec in report['recommendations']:
            st.write(f"• {rec}")
    
    # Risk Assessment
    with st.expander("⚠️ Risk Assessment"):
        risk = report['risk_assessment']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Validation Risk", risk['validation_risk'])
        
        with col2:
            st.metric("Data Consistency", risk['data_consistency'])
        
        with col3:
            st.metric("Verification Required", "Yes" if risk['verification_required'] else "No")
        
        with col4:
            st.metric("Critical Issues", risk['critical_issues'])

def display_detailed_report(report):
    """Display detailed validation report"""
    
    st.subheader("📋 Detailed Validation Report")
    
    # Report metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Institution:** {report['institution_name']}")
        st.write(f"**Validation Date:** {report['validation_date']}")
    
    with col2:
        st.write(f"**Total Metrics Compared:** {report['summary']['total_metrics_compared']}")
        st.write(f"**Overall Confidence:** {report['summary']['confidence_score']:.2%}")
    
    # Detailed breakdown
    st.write("### Detailed Metrics Breakdown")
    
    # Create detailed table
    detailed_data = []
    
    for category, counts in report['category_breakdown'].items():
        total = counts['matches'] + counts['mismatches'] + counts['missing']
        detailed_data.append({
            'Category': category.replace('_', ' ').title(),
            'Total Metrics': total,
            '✅ Matches': counts['matches'],
            '⚠️ Mismatches': counts['mismatches'],
            '❌ Missing': counts['missing'],
            'Accuracy': f"{(counts['matches'] / total * 100) if total > 0 else 0:.1f}%"
        })
    
    st.dataframe(pd.DataFrame(detailed_data), use_container_width=True)
    
    # Export option
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"validation_report_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

def display_discrepancies(results):
    """Display detailed discrepancies"""
    
    st.subheader("⚠️ Detailed Discrepancies")
    
    if not results['mismatches']:
        st.success("✅ No discrepancies found!")
        return
    
    # Group discrepancies by severity
    high_discrepancies = [m for m in results['mismatches'] if m['similarity'] < 0.5]
    medium_discrepancies = [m for m in results['mismatches'] if 0.5 <= m['similarity'] < 0.7]
    low_discrepancies = [m for m in results['mismatches'] if m['similarity'] >= 0.7]
    
    # High severity discrepancies
    if high_discrepancies:
        st.error("🔴 **Critical Discrepancies** (Similarity < 50%)")
        for discrepancy in high_discrepancies[:5]:  # Show top 5
            st.write(f"""
            - **{discrepancy['category'].replace('_', ' ').title()}: {discrepancy['metric'].replace('_', ' ')}**
              Document: {discrepancy['document_value']}
              Form: {discrepancy['form_value']}
              Similarity: {discrepancy['similarity']:.2%}
              Issue: {discrepancy.get('discrepancy', 'Major difference')}
            """)
    
    # Medium severity discrepancies
    if medium_discrepancies:
        st.warning("🟠 **Moderate Discrepancies** (50-70% Similarity)")
        for discrepancy in medium_discrepancies[:3]:
            st.write(f"""
            - **{discrepancy['category'].replace('_', ' ').title()}: {discrepancy['metric'].replace('_', ' ')}**
              Document: {discrepancy['document_value']}
              Form: {discrepancy['form_value']}
              Similarity: {discrepancy['similarity']:.2%}
            """)
    
    # Low severity discrepancies
    if low_discrepancies:
        st.info("🟡 **Minor Discrepancies** (>70% Similarity)")
        for discrepancy in low_discrepancies[:2]:
            st.write(f"""
            - **{discrepancy['category'].replace('_', ' ').title()}: {discrepancy['metric'].replace('_', ' ')}**
              Similarity: {discrepancy['similarity']:.2%}
            """)
    
    # Missing data
    if results['missing_in_document']:
        st.warning("📄 **Missing in Documents**")
        for missing in results['missing_in_document'][:3]:
            st.write(f"- {missing['category'].replace('_', ' ')}: {missing['metric'].replace('_', ' ')}")
            st.write(f"  Form value: {missing['form_value']}")

def export_validation_results(report):
    """Export validation results"""
    
    # Create downloadable JSON
    json_data = json.dumps(report, indent=2)
    
    # Create summary CSV
    summary_data = {
        'Institution': [report['institution_name']],
        'Validation_Date': [report['validation_date']],
        'Confidence_Score': [report['summary']['confidence_score']],
        'Match_Percentage': [report['summary']['overall_match_percentage']],
        'Total_Metrics': [report['summary']['total_metrics_compared']],
        'Matches': [report['summary']['matches']],
        'Mismatches': [report['summary']['mismatches']],
        'Validation_Risk': [report['risk_assessment']['validation_risk']]
    }
    
    df = pd.DataFrame(summary_data)
    csv_data = df.to_csv(index=False)
    
    # Offer both formats
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_data,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col2:
        st.download_button(
            label="📊 Download Summary CSV",
            data=csv_data,
            file_name=f"validation_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# To integrate with your main application, add this to the imports and navigation:
# In main.py, add to imports:
# from rag_score import create_rag_validation_dashboard

# In the navigation panel, add:
# "🔍 Document-Form Validation" to the app_mode options

# In the routing logic, add:
# elif app_mode == "🔍 Document-Form Validation":
#     create_rag_validation_dashboard(analyzer)
