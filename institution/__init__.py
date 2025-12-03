# institution/__init__.py
"""
Institution Module Package

This package contains all institution-specific functionality including:
- Authentication
- Dashboard
- Data submission forms
- Document management
- Submission tracking
"""

from institution.auth import create_institution_login
from institution.dashboard import create_institution_dashboard
from institution.forms import (
    create_institution_data_submission,
    create_systematic_data_submission_form
)
from institution.documents import create_institution_document_upload
from institution.submissions import (
    create_institution_submissions_view,
    create_institution_requirements_guide,
    create_institution_approval_workflow
)

__all__ = [
    'create_institution_login',
    'create_institution_dashboard',
    'create_institution_data_submission',
    'create_systematic_data_submission_form',
    'create_institution_document_upload',
    'create_institution_submissions_view',
    'create_institution_requirements_guide',
    'create_institution_approval_workflow'
]
