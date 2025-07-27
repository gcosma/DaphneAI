# app.py
# Simple Document Search - Upload, Search, Get Ranked Results

import streamlit as st
import re
from datetime import datetime

# Simple document processing
def process_file(uploaded_file):
    """Extract text from uploaded file"""
    filename = uploaded_file.name
    
    try:
        if filename.lower().endswith('.txt'):
            text = uploaded_file.getvalue().decode('utf-8')
        elif filename.lower().endswith('.pdf'):
            # Simple PDF processing
            try:
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            except ImportError:
                return {'filename': filename, 'error': 'Install pdfplumber: pip install pdfplumber'}
        elif filename.lower().endswith('.docx'):
            # Simple DOCX processing
            try:
                import docx
                from io import BytesIO
                doc = docx.Document(BytesIO(uploaded_file.getvalue()))
                text = '\n'.join([p.text for p in doc.paragraphs])
            except ImportError:
                return {'filename': filename, 'error': 'Install python-docx: pip install python-docx'}
        else:
            return {'filename': filename, 'error': 'Unsupported file type'}
        
        return {
            'filename': filename,
            'text': text,
            'word_count': len(text.split())
        }
    except Exception as e:
        return {'filename': filename, 'error': str(e)}

def search_documents(query, documents):
    """Search documents and return ranked results"""
    if not query or not documents:
        return []
    
    results = []
    query_lower = query.lower()
    query_words = query_lower.split()
    
    for doc in documents:
        if 'text' not in doc:
            continue
            
        text_lower = doc['text'].lower()
        score = 0
        
        # Exact phrase match (highest score)
        if query_lower in text_lower:
            score += 10
        
        # Individual word matches
        word_matches = sum(1 for word in query_words if word in text_lower)
        score += word_matches * 2
        
        # Word frequency bonus
        for word in query_words:
            score += text_lower.count(word) * 0.5
        
        if score > 0:
            # Find best snippet
            snippet = find_snippet(doc['text'], query, 200)
            
            results.append({
                'filename': doc['filename'],
                'score': score,
                'snippet': snippet,
                'word_count': doc.get('word_count', 0)
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def find_snippet(text, query, max_length=200):
    """Find the best snippet containing the query"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Find first occurrence of query
    index = text_lower.find(query_lower)
    if index == -1:
        # If exact phrase not found, find first word
        words = query_lower.split()
        for word in words:
            index = text_lower.find(word)
            if index != -1:
                break
    
    if index == -1:
        return text[:max_length] + "..."
    
    # Extract snippet around the found text
    start = max(0, index - max_length // 2)
    end = min(len(text), index + len(query) + max_length // 2)
    snippet = text[start:end]
    
    # Highlight the query in the snippet
    snippet = re.sub(f'({re.escape(query)})', r'**\1**', snippet, flags=re.IGNORECASE)
    
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")

# Streamlit App
st.set_page_config(page_title="Document Search", page_icon="🔍", layout="wide")

st.title("🔍 Simple Document Search")
st.write("Upload documents and search them - that's it!")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Upload section
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files", 
    type=['pdf', 'txt', 'docx'], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Files"):
        st.session_state.documents = []
        
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            st.write(f"Processing {file.name}...")
            doc = process_file(file)
            st.session_state.documents.append(doc)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.success(f"Processed {len(uploaded_files)} files!")

# Show loaded documents
if st.session_state.documents:
    st.write(f"**Loaded {len(st.session_state.documents)} documents:**")
    for doc in st.session_state.documents:
        if 'error' in doc:
            st.error(f"❌ {doc['filename']}: {doc['error']}")
        else:
            st.write(f"✅ {doc['filename']} ({doc['word_count']} words)")

# Search section
st.header("🔍 Search")
query = st.text_input("Enter your search query:", placeholder="Type what you're looking for...")

if query and st.session_state.documents:
    results = search_documents(query, st.session_state.documents)
    
    if results:
        st.write(f"**Found {len(results)} results for '{query}':**")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"{i}. {result['filename']} (Score: {result['score']:.1f})"):
                st.write(f"**Word count:** {result['word_count']}")
                st.write("**Relevant excerpt:**")
                st.markdown(result['snippet'])
    else:
        st.write("No results found.")

elif query:
    st.write("Please upload some documents first.")

# Sample data for testing
if not st.session_state.documents:
    if st.button("🎯 Load Sample Data for Testing"):
        st.session_state.documents = [
            {
                'filename': 'sample1.txt',
                'text': 'The government recommends implementing new healthcare policies to improve patient access and reduce waiting times.',
                'word_count': 16
            },
            {
                'filename': 'sample2.txt', 
                'text': 'This document discusses the importance of data security and privacy protection in government systems.',
                'word_count': 15
            }
        ]
        st.rerun()
