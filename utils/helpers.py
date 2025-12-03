import pandas as pd
import numpy as np
from datetime import datetime

def get_document_requirements_by_parameters(approval_type):
    """Get document requirements organized by parameters"""
    # This function is now in document_analysis.py, but we can keep it here too
    # for other modules to use if needed
    pass

def calculate_performance_percentile(score, inst_type, historical_data):
    """Calculate performance percentile within institution type"""
    type_data = historical_data[
        (historical_data['institution_type'] == inst_type) &
        (historical_data['year'] == 2023)
    ]
    
    if len(type_data) == 0:
        return 50.0
    
    return (type_data['performance_score'] < score).mean() * 100

def generate_approval_recommendation(performance_score):
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

def assess_risk_level(performance_score):
    """Assess institutional risk level"""
    if performance_score >= 8.0:
        return "Low Risk"
    elif performance_score >= 6.5:
        return "Medium Risk"
    elif performance_score >= 5.0:
        return "High Risk"
    else:
        return "Critical Risk"
