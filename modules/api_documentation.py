import streamlit as st
import requests
import json

def create_api_documentation():
    st.header("🌐 API Integration Portal")
    
    st.info("""
    **RESTful API for UGC/AICTE Institutional Analytics Platform**
    
    This API allows external systems, hackathon participants, and institutions to:
    - Access institutional performance data
    - Run AI-powered analytics
    - Get configuration parameters
    - Export data in multiple formats
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 API Documentation", 
        "🔑 API Keys", 
        "🧪 API Testing",
        "📊 Quick Integration"
    ])
    
    with tab1:
        create_api_documentation_tab()
    
    with tab2:
        create_api_keys_tab()
    
    with tab3:
        create_api_testing_tab()
    
    with tab4:
        create_integration_guide_tab()

def create_api_documentation_tab():
    """Create API documentation tab"""
    st.subheader("API Endpoints Overview")
    
    endpoints = [
        {
            "endpoint": "GET /institutions",
            "description": "Get list of institutions",
            "parameters": "year, institution_type, state, limit",
            "example_response": """{
    "institutions": [
        {
            "institution_id": "INST_0001",
            "institution_name": "University 001",
            "institution_type": "State University",
            "state": "Maharashtra",
            "year": 2023
        }
    ]
}"""
        },
        {
            "endpoint": "GET /institutions/{id}",
            "description": "Get institution performance data",
            "parameters": "start_year, end_year",
            "example_response": """{
    "institution_id": "INST_0001",
    "institution_name": "University 001",
    "performance_score": 8.5,
    "risk_level": "Low Risk",
    "approval_recommendation": "Full Approval - 5 Years"
}"""
        },
        {
            "endpoint": "GET /performance/ranking",
            "description": "Get performance ranking",
            "parameters": "year, institution_type, limit",
            "example_response": """{
    "rankings": [
        {
            "rank": 1,
            "institution_id": "INST_0001",
            "performance_score": 9.2
        }
    ]
}"""
        },
        {
            "endpoint": "POST /analysis/run",
            "description": "Run AI analysis",
            "parameters": "institution_id, analysis_type",
            "example_response": """{
    "analysis_id": "ANALYSIS_001",
    "status": "completed",
    "results": {
        "performance_score": 8.5,
        "risk_assessment": "Low Risk"
    }
}"""
        }
    ]
    
    for endpoint in endpoints:
        with st.expander(f"🌐 {endpoint['endpoint']}"):
            st.write(f"**Description:** {endpoint['description']}")
            st.write(f"**Parameters:** {endpoint['parameters']}")
            
            st.code(f"""
# Python Example
import requests

api_key = "your_api_key_here"
base_url = "https://api.ugc-sugam.gov.in/v1"
headers = {{"Authorization": f"Bearer {{api_key}}"}}

# Request for {endpoint['endpoint']}
response = requests.get(
    f"{{base_url}}{endpoint['endpoint'].replace('{id}', 'INST_0001')}",
    headers=headers,
    params={{"year": 2023}}
)

print(response.json())
            """, language="python")
            
            st.write("**Example Response:**")
            st.json(json.loads(endpoint['example_response']))

def create_api_keys_tab():
    """Create API keys tab"""
    st.subheader("🔑 API Access Keys")
    
    st.warning("⚠️ These keys are for demonstration. Use secure keys in production.")
    
    api_keys = {
        "ugc_admin": "ugc_api_key_2024_secure",
        "aictel_team": "aictel_api_2024_secure", 
        "hackathon_2024": "smart_india_hackathon_key",
        "institution_api": "institution_access_2024"
    }
    
    for role, key in api_keys.items():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**{role.replace('_', ' ').title()}**")
        with col2:
            st.code(key, language="text")
    
    st.info("""
    **Usage:**
    ```python
    headers = {
        "Authorization": "Bearer your_api_key_here"
    }
    ```
    """)
    
    # API key generation
    st.subheader("🔐 Generate New API Key")
    
    with st.form("generate_api_key"):
        api_name = st.text_input("API Key Name")
        permissions = st.multiselect(
            "Permissions",
            ["read:institutions", "write:institutions", "read:analysis", "write:analysis", "admin"]
        )
        expiry_days = st.slider("Expiry (days)", 1, 365, 30)
        
        if st.form_submit_button("Generate API Key"):
            if api_name:
                # In a real implementation, this would generate a secure key
                dummy_key = f"demo_key_{api_name.lower().replace(' ', '_')}_{expiry_days}d"
                st.success(f"API Key Generated: `{dummy_key}`")
                st.info("**Note:** This is a demo key. In production, use secure key generation.")
            else:
                st.error("Please enter an API key name")

def create_api_testing_tab():
    """Create API testing tab"""
    st.subheader("🧪 API Testing Interface")
    
    # Quick test interface
    endpoint = st.selectbox("Select Endpoint", [
        "/institutions",
        "/institutions/INST_0001", 
        "/performance/ranking",
        "/metrics",
        "/documents"
    ])
    
    api_key = st.selectbox("Select API Key", [
        "smart_india_hackathon_key",
        "ugc_api_key_2024_secure",
        "institution_access_2024"
    ])
    
    if st.button("🚀 Test API Endpoint"):
        with st.spinner("Testing API endpoint..."):
            # Simulate API call
            st.success("✅ API Test Successful!")
            
            # Show expected response structure
            if "/institutions" in endpoint and "INST" not in endpoint:
                st.json({
                    "institutions": [
                        {
                            "institution_id": "INST_0001",
                            "institution_name": "University 001",
                            "institution_type": "State University",
                            "state": "Maharashtra",
                            "year": 2023
                        }
                    ]
                })
            elif "/institutions/INST" in endpoint:
                st.json({
                    "institution_id": "INST_0001",
                    "institution_name": "University 001",
                    "performance_score": 8.5,
                    "risk_level": "Low Risk",
                    "approval_recommendation": "Full Approval - 5 Years"
                })
    
    # Advanced testing
    st.subheader("🔧 Advanced API Testing")
    
    with st.form("advanced_api_test"):
        custom_endpoint = st.text_input("Custom Endpoint", "/institutions")
        request_method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
        request_body = st.text_area("Request Body (JSON)", "{}", height=100)
        
        if st.form_submit_button("Send Request"):
            st.info("**Request Details:**")
            st.code(f"""
Method: {request_method}
Endpoint: {custom_endpoint}
Body: {request_body}
            """, language="json")
            
            st.success("✅ Request would be sent to API server")
            st.info("**Note:** This is a simulation. In production, this would make actual API calls.")

def create_integration_guide_tab():
    """Create integration guide tab"""
    st.subheader("📊 Quick Integration Guide")
    
    # Python SDK download
    python_sdk = """
# UGC/AICTE API Python Client
import requests

class UGCAnalyticsAPI:
    def __init__(self, api_key, base_url="https://api.ugc-sugam.gov.in/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_institutions(self, year=None, institution_type=None):
        params = {}
        if year: params['year'] = year
        if institution_type: params['institution_type'] = institution_type
        
        response = requests.get(
            f"{self.base_url}/institutions",
            headers=self.headers,
            params=params
        )
        return response.json()
    
    def get_performance(self, institution_id):
        response = requests.get(
            f"{self.base_url}/institutions/{institution_id}",
            headers=self.headers
        )
        return response.json()
    
    def run_analysis(self, institution_id, analysis_type="comprehensive"):
        data = {
            "institution_id": institution_id,
            "analysis_type": analysis_type
        }
        response = requests.post(
            f"{self.base_url}/analysis/run",
            headers=self.headers,
            json=data
        )
        return response.json()

# Usage example
api = UGCAnalyticsAPI(api_key="your_api_key")
institutions = api.get_institutions(year=2023)
print(f"Found {len(institutions)} institutions")
    """
    
    st.download_button(
        label="📥 Download Python SDK",
        data=python_sdk,
        file_name="ugc_api_client.py",
        mime="text/x-python"
    )
    
    # Integration examples
    st.subheader("💡 Integration Examples")
    
    examples_tabs = st.tabs(["Python", "JavaScript", "cURL"])
    
    with examples_tabs[0]:
        st.code("""
# Install required package
# pip install requests

import requests

# API Configuration
API_KEY = "your_api_key_here"
BASE_URL = "https://api.ugc-sugam.gov.in/v1"

# Get institutions
response = requests.get(
    f"{BASE_URL}/institutions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    params={"year": 2023, "limit": 10}
)

if response.status_code == 200:
    data = response.json()
    for institution in data['institutions']:
        print(f"{institution['institution_name']}: {institution['performance_score']}")
else:
    print(f"Error: {response.status_code}")
        """, language="python")
    
    with examples_tabs[1]:
        st.code("""
// Using fetch API in JavaScript
const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.ugc-sugam.gov.in/v1';

async function getInstitutions() {
    try {
        const response = await fetch(`${BASE_URL}/institutions?year=2023&limit=10`, {
            headers: {
                'Authorization': `Bearer ${API_KEY}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('Institutions:', data.institutions);
            return data;
        } else {
            console.error('Error:', response.status);
        }
    } catch (error) {
        console.error('Network error:', error);
    }
}

// Call the function
getInstitutions();
        """, language="javascript")
    
    with examples_tabs[2]:
        st.code("""
# Get institutions using cURL
curl -X GET "https://api.ugc-sugam.gov.in/v1/institutions?year=2023&limit=10" \
     -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json"

# Get specific institution
curl -X GET "https://api.ugc-sugam.gov.in/v1/institutions/INST_0001" \
     -H "Authorization: Bearer your_api_key_here"

# Run analysis
curl -X POST "https://api.ugc-sugam.gov.in/v1/analysis/run" \
     -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"institution_id": "INST_0001", "analysis_type": "comprehensive"}'
        """, language="bash")
    
    # Webhook setup
    st.subheader("🔔 Webhook Integration")
    
    with st.expander("Set up Webhooks"):
        st.write("""
        **Webhook Configuration:**
        
        1. **Endpoint URL:** Your server endpoint that will receive webhook events
        2. **Events to Subscribe:**
           - `institution.updated`
           - `analysis.completed`
           - `approval.status_changed`
           - `document.uploaded`
        
        3. **Secret Key:** For verifying webhook signatures
        
        **Example Webhook Payload:**
        ```json
        {
          "event": "institution.updated",
          "data": {
            "institution_id": "INST_0001",
            "performance_score": 8.5,
            "updated_at": "2024-01-15T10:30:00Z"
          },
          "signature": "sha256=..."
        }
        ```
        
        **Verification:**
        Verify the webhook signature to ensure it came from UGC/AICTE.
        """)
