import sqlite3
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

def init_database():
    """Initialize SQLite database for storing institutional data"""
    conn = sqlite3.connect('data/institutions.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create institutions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT UNIQUE,
            institution_name TEXT,
            year INTEGER,
            institution_type TEXT,
            state TEXT,
            established_year INTEGER,
            naac_grade TEXT,
            nirf_ranking INTEGER,
            student_faculty_ratio REAL,
            phd_faculty_ratio REAL,
            research_publications INTEGER,
            research_grants_amount REAL,
            patents_filed INTEGER,
            industry_collaborations INTEGER,
            digital_infrastructure_score REAL,
            library_volumes INTEGER,
            laboratory_equipment_score REAL,
            financial_stability_score REAL,
            compliance_score REAL,
            administrative_efficiency REAL,
            placement_rate REAL,
            higher_education_rate REAL,
            entrepreneurship_cell_score REAL,
            community_projects INTEGER,
            rural_outreach_score REAL,
            inclusive_education_index REAL,
            rusa_participation INTEGER,
            nmeict_participation INTEGER,
            fist_participation INTEGER,
            dst_participation INTEGER,
            performance_score REAL,
            approval_recommendation TEXT,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create other tables...
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS institution_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            document_name TEXT,
            document_type TEXT,
            file_path TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Pending',
            extracted_data TEXT,
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            analysis_type TEXT,
            extracted_data TEXT,
            ai_insights TEXT,
            confidence_score REAL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS institution_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            submission_type TEXT,
            submission_data TEXT,
            status TEXT DEFAULT 'Under Review',
            submitted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_by TEXT,
            review_date TIMESTAMP,
            review_comments TEXT,
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS institution_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            username TEXT UNIQUE,
            password_hash TEXT,
            contact_person TEXT,
            email TEXT,
            phone TEXT,
            role TEXT DEFAULT 'Institution',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
        )
    ''')
    
    conn.commit()
    return conn

def load_or_generate_data(conn):
    """Load data from database or generate sample data with 20×10 specification"""
    try:
        # Try to load from database
        df = pd.read_sql('SELECT * FROM institutions', conn)
        if len(df) > 0:
            # Verify the loaded data matches 20×10 specification
            unique_institutions = df['institution_id'].nunique()
            unique_years = df['year'].nunique()
            
            print(f"📊 Loaded data: {unique_institutions} institutions, {unique_years} years, {len(df)} records")
            
            # If data doesn't match 20×10, regenerate
            if unique_institutions != 20 or unique_years != 10:
                print("⚠️ Data doesn't match 20×10 specification. Regenerating...")
                df = generate_comprehensive_historical_data()
                df.to_sql('institutions', conn, if_exists='replace', index=False)
            
            return df
    except:
        pass
    
    # Generate sample data if database is empty
    print("🔄 Generating new 20×10 sample data...")
    sample_data = generate_comprehensive_historical_data()
    sample_data.to_sql('institutions', conn, if_exists='replace', index=False)
    return sample_data

def generate_comprehensive_historical_data() -> pd.DataFrame:
    """Generate comprehensive historical data for 20 institutions over 10 years"""
    np.random.seed(42)
    n_institutions = 20
    years_of_data = 10
    
    institutions_data = []
    
    for inst_id in range(1, n_institutions + 1):
        base_quality = np.random.uniform(0.3, 0.9)
        
        for year_offset in range(years_of_data):
            year = 2023 - year_offset
            inst_trend = base_quality + (year_offset * 0.02)
            
            # Generate realistic data
            naac_grades = ['A++', 'A+', 'A', 'B++', 'B+', 'B', 'C']
            naac_probs = [0.05, 0.10, 0.15, 0.25, 0.25, 0.15, 0.05]
            naac_grade = np.random.choice(naac_grades, p=naac_probs)
            
            if base_quality > 0.7 and np.random.random() < 0.8:
                nirf_rank = np.random.randint(1, 101)
            elif base_quality > 0.5 and np.random.random() < 0.5:
                nirf_rank = np.random.randint(101, 201)
            else:
                nirf_rank = None
            
            student_faculty_ratio = max(10, np.random.normal(20, 5))
            phd_faculty_ratio = np.random.beta(2, 2) * 0.6 + 0.3
            
            publications = max(0, int(np.random.poisson(inst_trend * 30)))
            research_grants = max(0, int(np.random.exponential(inst_trend * 500000)))
            patents = np.random.poisson(inst_trend * 3)
            
            digital_infrastructure_score = max(1, min(10, np.random.normal(7, 1.5)))
            library_volumes = max(1000, int(np.random.normal(20000, 10000)))
            
            financial_stability = max(1, min(10, np.random.normal(7.5, 1.2)))
            compliance_score = max(1, min(10, np.random.normal(8, 1)))
            
            placement_rate = max(40, min(98, np.random.normal(75, 10)))
            higher_education_rate = max(5, min(50, np.random.normal(20, 8)))
            
            community_projects = np.random.poisson(inst_trend * 8)
            
            # Calculate performance score
            faculty_count = max(1, np.random.randint(30, 150))
            
            performance_score = calculate_performance_score({
                'naac_grade': naac_grade,
                'nirf_ranking': nirf_rank,
                'student_faculty_ratio': student_faculty_ratio,
                'phd_faculty_ratio': phd_faculty_ratio,
                'publications_per_faculty': publications / faculty_count,
                'research_grants': research_grants,
                'digital_infrastructure': digital_infrastructure_score,
                'financial_stability': financial_stability,
                'placement_rate': placement_rate,
                'community_engagement': community_projects
            })
            
            institution_data = {
                'institution_id': f'INST_{inst_id:04d}',
                'institution_name': f'University/College {inst_id:03d}',
                'year': year,
                'institution_type': np.random.choice(['State University', 'Deemed University', 'Private University', 'Autonomous College'], p=[0.3, 0.2, 0.3, 0.2]),
                'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 'Kerala', 'Gujarat'], p=[0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1]),
                'established_year': np.random.randint(1950, 2015),
                
                # Academic Metrics
                'naac_grade': naac_grade,
                'nirf_ranking': nirf_rank,
                'student_faculty_ratio': round(student_faculty_ratio, 1),
                'phd_faculty_ratio': round(phd_faculty_ratio, 3),
                
                # Research Metrics
                'research_publications': publications,
                'research_grants_amount': research_grants,
                'patents_filed': patents,
                'industry_collaborations': np.random.poisson(inst_trend * 6),
                
                # Infrastructure Metrics
                'digital_infrastructure_score': round(digital_infrastructure_score, 1),
                'library_volumes': library_volumes,
                'laboratory_equipment_score': round(max(1, min(10, np.random.normal(7, 1.3))), 1),
                
                # Governance Metrics
                'financial_stability_score': round(financial_stability, 1),
                'compliance_score': round(compliance_score, 1),
                'administrative_efficiency': round(max(1, min(10, np.random.normal(7.2, 1.1))), 1),
                
                # Student Development Metrics
                'placement_rate': round(placement_rate, 1),
                'higher_education_rate': round(higher_education_rate, 1),
                'entrepreneurship_cell_score': round(max(1, min(10, np.random.normal(6.5, 1.5))), 1),
                
                # Social Impact Metrics
                'community_projects': community_projects,
                'rural_outreach_score': round(max(1, min(10, np.random.normal(6.8, 1.4))), 1),
                'inclusive_education_index': round(max(1, min(10, np.random.normal(7.5, 1.2))), 1),
                
                # Overall Performance
                'performance_score': round(performance_score, 2),
                'approval_recommendation': generate_approval_recommendation(performance_score),
                'risk_level': assess_risk_level(performance_score)
            }
            
            institutions_data.append(institution_data)
    
    df = pd.DataFrame(institutions_data)
    print(f"✅ Generated data for {df['institution_id'].nunique()} institutions across {df['year'].nunique()} years")
    print(f"📊 Total records: {len(df)} | Years: {df['year'].min()}-{df['year'].max()}")
    
    return df

def calculate_performance_score(metrics):
    """Calculate overall performance score based on weighted metrics"""
    score = 0
    
    # NAAC Grade scoring
    naac_scores = {'A++': 10, 'A+': 9, 'A': 8, 'B++': 7, 'B+': 6, 'B': 5, 'C': 4}
    naac_score = naac_scores.get(metrics.get('naac_grade'), 5)
    score += naac_score * 0.15
    
    # NIRF Ranking scoring (inverse)
    nirf_score = 0
    if metrics.get('nirf_ranking') and metrics['nirf_ranking'] <= 200:
        nirf_score = (201 - metrics['nirf_ranking']) / 200 * 10
    score += nirf_score * 0.10
    
    # Student-Faculty Ratio (lower is better)
    sf_ratio = metrics.get('student_faculty_ratio', 20)
    sf_ratio_score = max(0, 10 - max(0, sf_ratio - 15) / 3)
    score += sf_ratio_score * 0.10
    
    # PhD Faculty Ratio
    phd_score = metrics.get('phd_faculty_ratio', 0.6) * 10
    score += phd_score * 0.10
    
    # Research Publications per faculty
    pub_per_faculty = metrics.get('publications_per_faculty', 0)
    pub_score = min(10, pub_per_faculty * 3)
    score += pub_score * 0.10
    
    # Research Grants (log scale)
    grant_score = min(10, np.log1p(metrics.get('research_grants', 0) / 100000) * 2.5)
    score += grant_score * 0.10
    
    # Infrastructure
    infra_score = metrics.get('digital_infrastructure', 7)
    score += infra_score * 0.10
    
    # Financial Stability
    financial_score = metrics.get('financial_stability', 7.5)
    score += financial_score * 0.10
    
    # Placement Rate
    placement_score = metrics.get('placement_rate', 70) / 10
    score += placement_score * 0.10
    
    # Community Engagement
    community_score = min(10, metrics.get('community_engagement', 0) / 1.5)
    score += community_score * 0.05
    
    return min(10, score)

def generate_approval_recommendation(performance_score: float) -> str:
    """Generate approval recommendation based on performance score"""
    if performance_score >= 8.0:
        return "Full Approval - 5 Years"
    elif performance_score >= 7.0:
        return "Provisional Approval - 3 Years"
    elif performance_score >= 6.0:
        return "Conditional Approval - 1 Year"
    elif performance_score >= 5.0:
        return "Approval with Strict Monitoring - 1 Year"
    else:
        return "Rejection - Significant Improvements Required"

def assess_risk_level(performance_score: float) -> str:
    """Assess institutional risk level"""
    if performance_score >= 8.0:
        return "Low Risk"
    elif performance_score >= 6.5:
        return "Medium Risk"
    elif performance_score >= 5.0:
        return "High Risk"
    else:
        return "Critical Risk"

def hash_password(password: str) -> str:
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_institution_user(conn, institution_id: str, username: str, password: str, 
                          contact_person: str, email: str, phone: str):
    """Create new institution user account"""
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO institution_users 
            (institution_id, username, password_hash, contact_person, email, phone)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (institution_id, username, hash_password(password), 
              contact_person, email, phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_institution_user(conn, username: str, password: str):
    """Authenticate institution user"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT iu.*, i.institution_name 
        FROM institution_users iu 
        JOIN institutions i ON iu.institution_id = i.institution_id 
        WHERE iu.username = ? AND iu.is_active = 1
    ''', (username,))
    
    user = cursor.fetchone()
    if user:
        columns = [description[0] for description in cursor.description]
        user_dict = dict(zip(columns, user))
        
        password_hash = user_dict.get('password_hash')
        if password_hash and password_hash == hash_password(password):
            return {
                'institution_id': user_dict.get('institution_id'),
                'institution_name': user_dict.get('institution_name'),
                'username': user_dict.get('username'),
                'role': user_dict.get('role', 'Institution'),
                'contact_person': user_dict.get('contact_person', ''),
                'email': user_dict.get('email', '')
            }
    return None

def save_institution_submission(conn, institution_id: str, submission_type: str, 
                              submission_data: Dict):
    """Save institution submission data"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO institution_submissions 
        (institution_id, submission_type, submission_data)
        VALUES (?, ?, ?)
    ''', (institution_id, submission_type, json.dumps(submission_data)))
    conn.commit()

def get_institution_submissions(conn, institution_id: str) -> pd.DataFrame:
    """Get submissions for a specific institution"""
    return pd.read_sql('''
        SELECT * FROM institution_submissions 
        WHERE institution_id = ? 
        ORDER BY submitted_date DESC
    ''', conn, params=(institution_id,))

def save_uploaded_documents(conn, institution_id: str, uploaded_files: List, document_types: List[str]):
    """Save uploaded documents to database"""
    cursor = conn.cursor()
    for i, uploaded_file in enumerate(uploaded_files):
        cursor.execute('''
            INSERT INTO institution_documents (institution_id, document_name, document_type, status)
            VALUES (?, ?, ?, ?)
        ''', (institution_id, uploaded_file.name, document_types[i], 'Uploaded'))
    conn.commit()

def get_institution_documents(conn, institution_id: str) -> pd.DataFrame:
    """Get documents for a specific institution"""
    return pd.read_sql('''
        SELECT * FROM institution_documents 
        WHERE institution_id = ? 
        ORDER BY upload_date DESC
    ''', conn, params=(institution_id,))
