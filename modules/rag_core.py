# modules/rag_core.py
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

class InstitutionalDocument:
    """Enhanced document class for institutional data"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.embedding = None
        self.document_type = metadata.get('document_type', 'unknown')
        self.institution_id = metadata.get('institution_id', 'unknown')
        self.year = metadata.get('year', datetime.now().year)
    
    def to_dict(self):
        return {
            'content': self.content,
            'metadata': self.metadata,
            'document_type': self.document_type,
            'institution_id': self.institution_id,
            'year': self.year
        }

class SmartDocumentSplitter:
    """Intelligent document splitter for institutional documents"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_patterns = {
            'executive_summary': r'(Executive\s+Summary|Abstract|Overview)',
            'methodology': r'(Methodology|Approach|Procedure)',
            'results': r'(Results|Findings|Outcomes)',
            'recommendations': r'(Recommendations|Suggestions|Conclusions)',
            'financial': r'(Financial|Budget|Revenue|Expenditure)',
            'academic': r'(Academic|Curriculum|Program|Course)',
            'infrastructure': r'(Infrastructure|Facilities|Equipment)'
        }
    
    def split_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split document into logical sections"""
        #sections = []
        #current_section = ""
        #current_title = "Introduction"
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
        # Skip PDF annotations and URLs
            if line.startswith('<</Type') or line.startswith('http'):
                continue
            clean_lines.append(line)
    
        text = '\n'.join(clean_lines)
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new section
            section_found = False
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    if current_section:
                        sections.append({
                            'title': current_title,
                            'content': current_section,
                            'section_type': self._classify_section(current_title)
                        })
                    current_section = line + " "
                    current_title = line
                    section_found = True
                    break
            
            if not section_found:
                current_section += line + " "
        
        # Add the last section
        if current_section:
            sections.append({
                'title': current_title,
                'content': current_section,
                'section_type': self._classify_section(current_title)
            })
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify section by content"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['financial', 'budget', 'revenue', 'expense']):
            return 'financial'
        elif any(word in title_lower for word in ['academic', 'curriculum', 'program', 'course']):
            return 'academic'
        elif any(word in title_lower for word in ['result', 'finding', 'outcome', 'achievement']):
            return 'results'
        elif any(word in title_lower for word in ['infrastructure', 'facility', 'equipment', 'lab']):
            return 'infrastructure'
        elif any(word in title_lower for word in ['method', 'approach', 'procedure', 'process']):
            return 'methodology'
        elif any(word in title_lower for word in ['recommend', 'suggest', 'conclusion', 'future']):
            return 'recommendations'
        else:
            return 'general'
    
    def chunk_section(self, section_content: str, section_type: str) -> List[str]:
        """Create intelligent chunks based on section type"""
        if section_type == 'financial':
            # Keep financial tables together
            chunks = self._chunk_financial_content(section_content)
        elif section_type == 'academic':
            # Keep course descriptions together
            chunks = self._chunk_academic_content(section_content)
        else:
            # General chunking
            chunks = self._chunk_general_content(section_content)
        
        return chunks
    
    def _chunk_financial_content(self, text: str) -> List[str]:
        """Special chunking for financial tables"""
        # Look for table patterns
        table_pattern = r'(\b\d[\d,.]*\b\s+)+'
        tables = re.findall(table_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for table in tables:
            if len(current_chunk) + len(table) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = table
            else:
                current_chunk += " " + table
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else self._chunk_general_content(text)
    
    def _chunk_academic_content(self, text: str) -> List[str]:
        """Special chunking for academic content"""
        # Split by course/program headings
        course_pattern = r'([A-Z]{2,4}\s*\d{3,4}[\s:-]+[A-Za-z].+?)(?=[A-Z]{2,4}\s*\d{3,4}|$)'
        courses = re.findall(course_pattern, text, re.DOTALL)
        
        if courses:
            return [course.strip() for course in courses if course.strip()]
        
        return self._chunk_general_content(text)
    
    def _chunk_general_content(self, text: str) -> List[str]:
        """General text chunking with sentence awareness"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class ComplianceChecker:
    """Check documents against regulatory requirements"""
    
    def __init__(self):
        self.requirements = self._load_compliance_requirements()
    
    def _load_compliance_requirements(self):
        """Load UGC/AICTE compliance requirements"""
        return {
            'naac': {
                'mandatory': ['SSR', 'IQAC Report', 'Annual Report'],
                'metrics': ['CGPA', 'Grade', 'Criteria Scores'],
                'thresholds': {'CGPA': 2.5, 'Grade': 'A'}
            },
            'aicte': {
                'mandatory': ['Approval Process Handbook', 'Faculty Details', 'Infrastructure Report'],
                'metrics': ['Faculty-Student Ratio', 'Lab Equipment', 'Library Collection'],
                'thresholds': {'Faculty-Student Ratio': '1:15'}
            },
            'financial': {
                'mandatory': ['Audit Report', 'Balance Sheet', 'Income Statement'],
                'metrics': ['Liquidity Ratio', 'Debt Ratio', 'Operating Margin'],
                'thresholds': {'Liquidity Ratio': 1.5}
            }
        }
    
    def check_document_compliance(self, document_content: str, doc_type: str) -> Dict[str, Any]:
        """Check if document meets compliance requirements"""
        results = {
            'compliance_score': 0,
            'missing_elements': [],
            'found_elements': [],
            'warnings': [],
            'recommendations': []
        }
        
        if doc_type not in self.requirements:
            return results
        
        req = self.requirements[doc_type]
        
        # Check for mandatory sections
        for element in req['mandatory']:
            if re.search(r'\b' + re.escape(element) + r'\b', document_content, re.IGNORECASE):
                results['found_elements'].append(element)
            else:
                results['missing_elements'].append(element)
                results['warnings'].append(f"Missing mandatory element: {element}")
        
        # Check metrics
        for metric in req['metrics']:
            # Look for metric values
            pattern = rf'{metric}[\s:]*([\d.,]+)'
            matches = re.findall(pattern, document_content, re.IGNORECASE)
            if matches:
                results['found_elements'].append(f"{metric}: {matches[-1]}")
        
        # Calculate compliance score
        total_elements = len(req['mandatory']) + len(req['metrics'])
        found_elements = len(results['found_elements'])
        if total_elements > 0:
            results['compliance_score'] = (found_elements / total_elements) * 100
        
        # Generate recommendations
        if results['missing_elements']:
            results['recommendations'].append(
                f"Add missing sections: {', '.join(results['missing_elements'])}"
            )
        
        return results

class InstitutionalRAGSystem:
    """Main RAG system for institutional document analysis"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.documents = []
        self.splitter = SmartDocumentSplitter()
        self.compliance_checker = ComplianceChecker()
        self.vector_store = SimpleVectorStore()
        self.document_index = {}
        
    def process_institutional_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete institutional document"""
        
        # Create document object
        doc = InstitutionalDocument(content, metadata)
        
        # Split into sections
        sections = self.splitter.split_by_sections(content)
        
        # Process each section
        processed_sections = []
        for section in sections:
            # Create chunks
            chunks = self.splitter.chunk_section(section['content'], section['section_type'])
            
            # Check compliance for relevant sections
            if section['section_type'] in ['financial', 'academic', 'results']:
                compliance = self.compliance_checker.check_document_compliance(
                    section['content'], 
                    section['section_type']
                )
                section['compliance'] = compliance
            
            processed_sections.append({
                **section,
                'chunks': chunks,
                'chunk_count': len(chunks)
            })
        
        # Store document
        self.documents.append(doc)
        doc_id = len(self.documents) - 1
        self.document_index[doc_id] = {
            'institution_id': metadata.get('institution_id'),
            'document_type': metadata.get('document_type'),
            'sections': [s['section_type'] for s in processed_sections],
            'total_chunks': sum(s['chunk_count'] for s in processed_sections)
        }
        
        # Extract key metrics
        metrics = self._extract_metrics(content, metadata.get('document_type'))
        
        return {
            'document_id': doc_id,
            'institution_id': metadata.get('institution_id'),
            'document_type': metadata.get('document_type'),
            'sections': processed_sections,
            'metrics': metrics,
            'total_sections': len(sections),
            'total_chunks': sum(s['chunk_count'] for s in processed_sections)
        }
    
    def _extract_metrics(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract key performance metrics from document"""
        metrics = {}
        
        # GPA/Grade extraction
        grade_patterns = [
            r'CGPA[\s:]*([\d.]+)',
            r'Grade[\s:]*([A-Z+]+)',
            r'Score[\s:]*([\d.]+)\s*/\s*[\d.]+'
        ]
        
        for pattern in grade_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metrics['assessment_score'] = matches[-1]
                break
        
        # Financial metrics
        financial_patterns = {
            'revenue': r'Revenue[\s:]*([\d,.]+)',
            'expenditure': r'Expenditure[\s:]*([\d,.]+)',
            'profit': r'Profit[\s:]*([\d,.]+)',
            'funding': r'Funding[\s:]*([\d,.]+)'
        }
        
        for metric, pattern in financial_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Clean number
                clean_num = matches[-1].replace(',', '')
                if '.' in clean_num:
                    metrics[metric] = float(clean_num)
                else:
                    metrics[metric] = int(clean_num)
        
        # Academic metrics
        academic_patterns = {
            'students': r'Students[\s:]*([\d,]+)',
            'faculty': r'Faculty[\s:]*([\d,]+)',
            'publications': r'Publications[\s:]*([\d,]+)',
            'placements': r'Placements[\s:]*([\d,]+)'
        }
        
        for metric, pattern in academic_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                clean_num = matches[-1].replace(',', '')
                metrics[metric] = int(clean_num)
        
        return metrics
    
    def query_institution(self, institution_id: str, question: str) -> Dict[str, Any]:
        """Query all documents for a specific institution"""
        
        # Find all documents for this institution
        institution_docs = []
        for doc_id, info in self.document_index.items():
            if info['institution_id'] == institution_id:
                institution_docs.append(self.documents[doc_id])
        
        if not institution_docs:
            return {
                'institution_id': institution_id,
                'found': False,
                'message': 'No documents found for this institution'
            }
        
        # Search across all documents
        all_results = []
        for doc in institution_docs:
            # Simple keyword search for now
            if question.lower() in doc.content.lower():
                # Find context around the match
                context_start = max(0, doc.content.lower().find(question.lower()) - 200)
                context_end = min(len(doc.content), context_start + 400)
                context = doc.content[context_start:context_end]
                
                all_results.append({
                    'document_type': doc.document_type,
                    'year': doc.year,
                    'context': context,
                    'metadata': doc.metadata
                })
        
        # Generate summary insights
        summary = self._generate_insights(institution_docs, question)
        
        return {
            'institution_id': institution_id,
            'found': len(all_results) > 0,
            'results': all_results,
            'summary': summary,
            'total_documents': len(institution_docs),
            'document_types': list(set(doc.document_type for doc in institution_docs))
        }
    
    def _generate_insights(self, documents: List[InstitutionalDocument], question: str) -> str:
        """Generate insights from documents"""
        
        # Extract key metrics from all documents
        all_metrics = []
        for doc in documents:
            metrics = self._extract_metrics(doc.content, doc.document_type)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return "No quantitative metrics found in documents."
        
        # Simple insight generation based on question
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['financial', 'revenue', 'budget', 'money']):
            revenues = [m.get('revenue', 0) for m in all_metrics if 'revenue' in m]
            if revenues:
                avg_revenue = sum(revenues) / len(revenues)
                return f"Average annual revenue: ₹{avg_revenue:,.2f}"
        
        elif any(word in question_lower for word in ['student', 'faculty', 'ratio']):
            students = [m.get('students', 0) for m in all_metrics if 'students' in m]
            faculty = [m.get('faculty', 0) for m in all_metrics if 'faculty' in m]
            if students and faculty:
                avg_ratio = sum(students) / sum(faculty) if sum(faculty) > 0 else 0
                return f"Average student-faculty ratio: {avg_ratio:.1f}:1"
        
        elif any(word in question_lower for word in ['publication', 'research']):
            publications = [m.get('publications', 0) for m in all_metrics if 'publications' in m]
            if publications:
                total_pubs = sum(publications)
                return f"Total research publications: {total_pubs}"
        
        return f"Found {len(all_metrics)} documents with quantitative data. Ask specific questions about financials, academics, or research."

    def analyze_institution_performance(self, institution_id: str) -> Dict[str, Any]:
        """Comprehensive performance analysis from documents"""
        
        institution_docs = []
        for doc_id, info in self.document_index.items():
            if info['institution_id'] == institution_id:
                institution_docs.append(self.documents[doc_id])
        
        if not institution_docs:
            return {'error': 'No documents found'}
        
        analysis = {
            'institution_id': institution_id,
            'total_documents': len(institution_docs),
            'document_types': {},
            'years_covered': set(),
            'key_findings': [],
            'compliance_status': {},
            'risk_indicators': []
        }
        
        # Analyze each document type
        for doc in institution_docs:
            doc_type = doc.document_type
            if doc_type not in analysis['document_types']:
                analysis['document_types'][doc_type] = 0
            analysis['document_types'][doc_type] += 1
            
            analysis['years_covered'].add(doc.year)
            
            # Check compliance
            compliance = self.compliance_checker.check_document_compliance(
                doc.content, 
                doc_type
            )
            analysis['compliance_status'][doc_type] = compliance
            
            # Look for risk indicators
            risk_keywords = ['deficit', 'decline', 'decrease', 'insufficient', 'non-compliance', 'violation']
            for keyword in risk_keywords:
                if keyword in doc.content.lower():
                    analysis['risk_indicators'].append({
                        'document_type': doc_type,
                        'year': doc.year,
                        'indicator': keyword,
                        'context': doc.content[doc.content.lower().find(keyword)-100:doc.content.lower().find(keyword)+100]
                    })
        
        # Generate overall assessment
        compliance_scores = [c['compliance_score'] for c in analysis['compliance_status'].values()]
        if compliance_scores:
            analysis['overall_compliance'] = sum(compliance_scores) / len(compliance_scores)
        
        analysis['years_covered'] = sorted(list(analysis['years_covered']))
        
        return analysis

# Keep your existing SimpleVectorStore (slightly enhanced)
class SimpleVectorStore:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = np.array([])
        self.metadata = []
    
    def add_document(self, text: str, metadata: dict = None):
        self.documents.append(text)
        self.metadata.append(metadata or {})
        
        # If we have an embedding model, create embedding
        if self.embedding_model:
            embedding = self.embedding_model.encode([text])[0]
            if not self.embeddings.size:
                self.embeddings = np.array([embedding])
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7):
        if not self.embeddings.size or self.embedding_model is None:
            # Fallback to keyword search
            results = []
            for i, doc in enumerate(self.documents):
                if query.lower() in doc.lower():
                    results.append({
                        'document': doc[:500] + '...',
                        'metadata': self.metadata[i],
                        'similarity': 1.0
                    })
            return results[:k]
        
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        results = []
        for idx in np.argsort(similarities)[-k:][::-1]:
            if similarities[idx] >= threshold:
                results.append({
                    'document': self.documents[idx][:500] + '...',
                    'metadata': self.metadata[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
