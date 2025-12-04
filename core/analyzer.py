import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional
from models.data_extractor import RAGDataExtractor
from modules.rag_core import InstitutionalRAGSystem, InstitutionalDocument
from core.database import (
    init_database, 
    load_or_generate_data, 
    calculate_performance_score, 
    generate_approval_recommendation, 
    assess_risk_level,
    create_institution_user,
    authenticate_institution_user,
    save_institution_submission,
    get_institution_submissions,
    save_uploaded_documents,
    get_institution_documents,
    hash_password
)

class InstitutionalAIAnalyzer:
    def __init__(self):
        self.conn = init_database()
        self.historical_data = load_or_generate_data(self.conn)
        self.performance_metrics = self.define_performance_metrics()
        self.document_requirements = self.define_document_requirements()
        self.rag_system = InstitutionalRAGSystem(self)
        
        # Initialize RAG with progress indication
        with st.spinner("🔄 Initializing AI Document Analysis System..."):
            try:
                if 'rag_initialized' not in st.session_state:
                    st.session_state.rag_initialized = False
                
                if not st.session_state.rag_initialized:
                    with st.spinner("🔄 Initializing AI Document Analysis System..."):
                        self.rag_extractor = RAGDataExtractor()
                        st.session_state.rag_initialized = True
                        
                        if hasattr(self.rag_extractor, 'embedding_model') and self.rag_extractor.embedding_model is not None:
                            st.success("✅ AI Document Analysis System Ready (Full Features)")
                        else:
                            st.info("ℹ️ AI Document Analysis System Ready (Basic Features - Text Extraction & Pattern Matching)")
            except Exception as e:
                st.warning(f"⚠️ AI System using basic mode: {e}")
                self.rag_extractor = RAGDataExtractor()
                st.session_state.rag_initialized = True
        
        self.create_dummy_institution_users()
        # Initialize report generator
        try:
            from modules.pdf_reports import PDFReportGenerator
            self.report_generator = PDFReportGenerator(self)
        except ImportError as e:
            st.warning(f"PDF Report Generator not available: {e}")
            self.report_generator = None

        from modules.rag_data_management import initialize_rag_for_analyzer
        self = initialize_rag_for_analyzer(self)
    
    def define_performance_metrics(self) -> Dict[str, Dict]:
        """Define key performance indicators for institutional evaluation"""
        return {
            "academic_excellence": {
                "weight": 0.25,
                "sub_metrics": {
                    "naac_grade": 0.30,
                    "nirf_ranking": 0.25,
                    "student_faculty_ratio": 0.20,
                    "phd_faculty_ratio": 0.15,
                    "curriculum_innovation": 0.10
                }
            },
            "research_innovation": {
                "weight": 0.20,
                "sub_metrics": {
                    "publications_per_faculty": 0.30,
                    "research_grants": 0.25,
                    "patents_filed": 0.20,
                    "conferences_organized": 0.15,
                    "industry_collaborations": 0.10
                }
            },
            "infrastructure_facilities": {
                "weight": 0.15,
                "sub_metrics": {
                    "campus_area": 0.25,
                    "digital_infrastructure": 0.25,
                    "library_resources": 0.20,
                    "laboratory_equipment": 0.20,
                    "hostel_facilities": 0.10
                }
            },
            "governance_administration": {
                "weight": 0.15,
                "sub_metrics": {
                    "financial_stability": 0.30,
                    "administrative_efficiency": 0.25,
                    "compliance_record": 0.25,
                    "grievance_redressal": 0.20
                }
            },
            "student_development": {
                "weight": 0.15,
                "sub_metrics": {
                    "placement_rate": 0.35,
                    "higher_education_rate": 0.20,
                    "entrepreneurship_cell": 0.15,
                    "extracurricular_activities": 0.15,
                    "alumni_network": 0.15
                }
            },
            "social_impact": {
                "weight": 0.10,
                "sub_metrics": {
                    "community_engagement": 0.30,
                    "rural_outreach": 0.25,
                    "inclusive_education": 0.25,
                    "environmental_initiatives": 0.20
                }
            }
        }
    
    def define_document_requirements(self) -> Dict[str, Dict]:
        """Define document requirements for different approval types"""
        return {
            "new_approval": {
                "mandatory": [
                    "affidavit_legal_status", "land_documents", "building_plan_approval",
                    "infrastructure_details", "financial_solvency_certificate",
                    "faculty_recruitment_plan", "academic_curriculum", "governance_structure"
                ],
                "supporting": [
                    "feasibility_report", "market_demand_analysis", "five_year_development_plan",
                    "industry_partnerships", "research_facilities_plan"
                ]
            },
            "renewal_approval": {
                "mandatory": [
                    "previous_approval_letters", "annual_reports", "financial_audit_reports",
                    "faculty_student_data", "infrastructure_utilization", "academic_performance"
                ],
                "supporting": [
                    "naac_accreditation", "nirf_data", "research_publications",
                    "placement_records", "social_impact_reports"
                ]
            },
            "expansion_approval": {
                "mandatory": [
                    "current_status_report", "expansion_justification", "additional_infrastructure",
                    "enhanced_faculty_plan", "financial_viability", "market_analysis"
                ],
                "supporting": [
                    "stakeholder_feedback", "alumni_support", "industry_demand",
                    "government_schemes_participation"
                ]
            }
        }
    
    def create_dummy_institution_users(self):
        """Create dummy institution users for testing"""
        dummy_users = [
            {
                'institution_id': 'INST_0001',
                'username': 'inst001_admin',
                'password': 'password123',
                'contact_person': 'Dr. Rajesh Kumar',
                'email': 'rajesh.kumar@university001.edu.in',
                'phone': '+91-9876543210'
            },
            {
                'institution_id': 'INST_0002',
                'username': 'inst002_registrar',
                'password': 'testpass456',
                'contact_person': 'Ms. Priya Sharma',
                'email': 'priya.sharma@college002.edu.in',
                'phone': '+91-8765432109'
            },
            {
                'institution_id': 'INST_0003',
                'username': 'inst003_director',
                'password': 'demo789',
                'contact_person': 'Prof. Amit Patel',
                'email': 'amit.patel@university003.edu.in',
                'phone': '+91-7654321098'
            }
        ]
        
        for user_data in dummy_users:
            try:
                create_institution_user(
                    self.conn,
                    user_data['institution_id'],
                    user_data['username'],
                    user_data['password'],
                    user_data['contact_person'],
                    user_data['email'],
                    user_data['phone']
                )
                print(f"Created user: {user_data['username']}")
            except Exception as e:
                print(f"Error creating user {user_data['username']}: {e}")
    
    # Wrapper methods for database operations
    def authenticate_institution_user(self, username: str, password: str):
        """Authenticate institution user"""
        return authenticate_institution_user(self.conn, username, password)
    
    def create_institution_user(self, institution_id: str, username: str, password: str, 
                              contact_person: str, email: str, phone: str):
        """Create new institution user account"""
        return create_institution_user(self.conn, institution_id, username, password, 
                                     contact_person, email, phone)
    
    def save_institution_submission(self, institution_id: str, submission_type: str, 
                                  submission_data: Dict):
        """Save institution submission data"""
        save_institution_submission(self.conn, institution_id, submission_type, submission_data)
    
    def get_institution_submissions(self, institution_id: str) -> pd.DataFrame:
        """Get submissions for a specific institution"""
        return get_institution_submissions(self.conn, institution_id)
    
    def save_uploaded_documents(self, institution_id: str, uploaded_files: List, document_types: List[str]):
        """Save uploaded documents to database"""
        save_uploaded_documents(self.conn, institution_id, uploaded_files, document_types)
    
    def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
        """Get documents for a specific institution"""
        return get_institution_documents(self.conn, institution_id)
    
    def analyze_documents_with_rag(self, institution_id: str, uploaded_files: List) -> Dict[str, Any]:
        """Analyze uploaded documents using available analysis methods"""
        return self.rag_extractor.analyze_documents_with_rag(institution_id, uploaded_files)
    
    # Add other methods that were in the original analyzer class
    def generate_comprehensive_report(self, institution_id: str) -> Dict[str, Any]:
        """Generate comprehensive AI analysis report for an institution"""
        inst_data = self.historical_data[
            self.historical_data['institution_id'] == institution_id
        ]
        
        if inst_data.empty:
            return {"error": "Institution not found"}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        historical_trend = inst_data.groupby('year')['performance_score'].mean()
        
        if len(historical_trend) > 1:
            if historical_trend.iloc[-1] > historical_trend.iloc[-2]:
                trend_analysis = "Improving"
            elif historical_trend.iloc[-1] == historical_trend.iloc[-2]:
                trend_analysis = "Stable"
            else:
                trend_analysis = "Declining"
        else:
            trend_analysis = "Insufficient Data"
        
        return {
            "institution_info": {
                "name": latest_data['institution_name'],
                "type": latest_data['institution_type'],
                "state": latest_data['state'],
                "established": latest_data['established_year']
            },
            "performance_analysis": {
                "current_score": latest_data['performance_score'],
                "historical_trend": historical_trend.to_dict(),
                "trend_analysis": trend_analysis,
                "approval_recommendation": latest_data['approval_recommendation'],
                "risk_level": latest_data['risk_level']
            },
            "strengths": self.identify_strengths(latest_data),
            "weaknesses": self.identify_weaknesses(latest_data),
            "ai_recommendations": self.generate_ai_recommendations(latest_data)
        }
    
    def identify_strengths(self, institution_data: pd.Series) -> List[str]:
        """Identify institutional strengths"""
        strengths = []
        
        if institution_data['naac_grade'] in ['A++', 'A+', 'A']:
            strengths.append(f"Excellent NAAC Accreditation: {institution_data['naac_grade']}")
        
        if institution_data['placement_rate'] > 80:
            strengths.append(f"Strong Placement Record: {institution_data['placement_rate']:.1f}%")
        
        if institution_data['research_publications'] > 100:
            strengths.append(f"Robust Research Output: {institution_data['research_publications']} publications")
        
        return strengths
    
    def identify_weaknesses(self, institution_data: pd.Series) -> List[str]:
        """Identify institutional weaknesses"""
        weaknesses = []
        
        if institution_data['student_faculty_ratio'] > 25:
            weaknesses.append(f"High Student-Faculty Ratio: {institution_data['student_faculty_ratio']:.1f}")
        
        if institution_data['placement_rate'] < 65:
            weaknesses.append(f"Low Placement Rate: {institution_data['placement_rate']:.1f}%")
        
        if institution_data['research_publications'] < 20:
            weaknesses.append(f"Inadequate Research Output: {institution_data['research_publications']} publications")
        
        return weaknesses
    
    def generate_ai_recommendations(self, institution_data: pd.Series) -> List[str]:
        """Generate AI-powered improvement recommendations"""
        recommendations = []
        
        if institution_data['student_faculty_ratio'] > 25:
            recommendations.append("Recruit additional faculty members to improve student-faculty ratio")
        
        if institution_data['placement_rate'] < 70:
            recommendations.append("Strengthen industry partnerships and career development programs")
        
        if institution_data['research_publications'] < 50:
            recommendations.append("Establish research promotion policy and faculty development programs")
        
        return recommendations

def get_institution_submissions(self, institution_id: str) -> pd.DataFrame:
    """Get submissions for a specific institution"""
    try:
        query = '''
            SELECT * FROM institution_submissions 
            WHERE institution_id = ? 
            ORDER BY submitted_date DESC
        '''
        return pd.read_sql(query, self.conn, params=(institution_id,))
    except Exception as e:
        print(f"Error getting submissions: {e}")
        return pd.DataFrame()

def save_institution_submission(self, institution_id: str, submission_type: str, 
                              submission_data: Dict):
    """Save institution submission data"""
    try:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO institution_submissions 
            (institution_id, submission_type, submission_data, status)
            VALUES (?, ?, ?, ?)
        ''', (institution_id, submission_type, 
              json.dumps(submission_data), 'Under Review'))
        self.conn.commit()
        
        # Update form submission count
        if 'form_submission_count' in st.session_state:
            st.session_state.form_submission_count += 1
            
        return True
    except Exception as e:
        print(f"Error saving submission: {e}")
        return False

def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
    """Get documents for a specific institution"""
    try:
        query = '''
            SELECT * FROM institution_documents 
            WHERE institution_id = ? 
            ORDER BY upload_date DESC
        '''
        return pd.read_sql(query, self.conn, params=(institution_id,))
    except Exception as e:
        print(f"Error getting documents: {e}")
        return pd.DataFrame()

def save_uploaded_documents(self, institution_id: str, uploaded_files: List, 
                          document_types: List[str]):
    """Save uploaded documents to database"""
    try:
        cursor = self.conn.cursor()
        for i, uploaded_file in enumerate(uploaded_files):
            # Extract filename and type
            filename = uploaded_file.name
            doc_type = document_types[i] if i < len(document_types) else 'other'
            
            cursor.execute('''
                INSERT INTO institution_documents 
                (institution_id, document_name, document_type, status)
                VALUES (?, ?, ?, ?)
            ''', (institution_id, filename, doc_type, 'Uploaded'))
        
        self.conn.commit()
        return True
    except Exception as e:
        print(f"Error saving documents: {e}")
        return False

def analyze_document_sufficiency(self, file_names: List[str], approval_type: str) -> Dict:
    """Analyze document sufficiency for approval type"""
    try:
        # Get requirements
        requirements = self.document_requirements.get(approval_type, {})
        
        mandatory_docs = requirements.get('mandatory', [])
        supporting_docs = requirements.get('supporting', [])
        
        # Count uploaded mandatory documents
        uploaded_mandatory = 0
        for doc in mandatory_docs:
            if any(doc.lower() in name.lower() for name in file_names):
                uploaded_mandatory += 1
        
        # Count uploaded supporting documents
        uploaded_supporting = 0
        for doc in supporting_docs:
            if any(doc.lower() in name.lower() for name in file_names):
                uploaded_supporting += 1
        
        # Calculate sufficiency percentages
        mandatory_sufficiency = (uploaded_mandatory / len(mandatory_docs) * 100) if mandatory_docs else 0
        overall_sufficiency = ((uploaded_mandatory + uploaded_supporting) / 
                              (len(mandatory_docs) + len(supporting_docs)) * 100) if (mandatory_docs and supporting_docs) else mandatory_sufficiency
        
        # Identify missing mandatory documents
        missing_mandatory = []
        for doc in mandatory_docs:
            if not any(doc.lower() in name.lower() for name in file_names):
                missing_mandatory.append(doc)
        
        # Generate recommendations
        recommendations = self.generate_document_recommendations(mandatory_sufficiency)
        
        return {
            'mandatory_sufficiency': mandatory_sufficiency,
            'overall_sufficiency': overall_sufficiency,
            'uploaded_mandatory': uploaded_mandatory,
            'uploaded_supporting': uploaded_supporting,
            'total_mandatory': len(mandatory_docs),
            'total_supporting': len(supporting_docs),
            'missing_mandatory': missing_mandatory,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"Error analyzing document sufficiency: {e}")
        return {
            'mandatory_sufficiency': 0,
            'overall_sufficiency': 0,
            'uploaded_mandatory': 0,
            'uploaded_supporting': 0,
            'total_mandatory': 0,
            'total_supporting': 0,
            'missing_mandatory': [],
            'recommendations': ["Error analyzing documents"]
        }

def generate_document_recommendations(self, mandatory_sufficiency: float) -> List[str]:
    """Generate recommendations based on document sufficiency"""
    recommendations = []
    
    if mandatory_sufficiency < 50:
        recommendations.append("❌ Critical: Upload all mandatory documents immediately")
        recommendations.append("⚠️ Application cannot proceed without mandatory documents")
    elif mandatory_sufficiency < 80:
        recommendations.append("📝 Important: Complete missing mandatory documents")
        recommendations.append("💡 Consider uploading supporting documents for better assessment")
    elif mandatory_sufficiency < 100:
        recommendations.append("✅ Good: Complete remaining mandatory documents")
        recommendations.append("🌟 Upload supporting documents for enhanced evaluation")
    else:
        recommendations.append("🎉 Excellent: All mandatory documents uploaded")
        recommendations.append("📊 Submit supporting documents for comprehensive assessment")
    
    return recommendations






