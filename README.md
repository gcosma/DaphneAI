# 🚀 Advanced Search Engine Setup Guide

## Quick Start

### 1. Install Dependencies

**Basic Installation (Smart Search only):**
```bash
pip install streamlit pandas PyPDF2 pdfplumber python-docx python-dotenv
```

**Full Installation (RAG + Smart Search):**
```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Features Overview

### 🔍 Search Modes

1. **🤖 RAG Semantic Search** - AI-powered meaning-based search
   - Uses sentence transformers for semantic understanding
   - Finds conceptually similar content, not just keyword matches
   - Best for: Finding related concepts, understanding context

2. **🔍 Smart Pattern Search** - Advanced keyword matching
   - Intelligent scoring with multiple factors
   - Fast and reliable for exact terms
   - Best for: Finding specific terms, phrases, or patterns

3. **🔄 Hybrid Search** - Combines RAG + Smart Search
   - Gets best of both approaches
   - Re-ranks results for optimal relevance
   - Best for: Maximum accuracy and coverage

4. **🎯 Fuzzy Search** - Typo-tolerant search
   - Uses TF-IDF for fuzzy matching
   - Handles misspellings and variations
   - Best for: When you're not sure of exact spelling

### 📌 Recommendation ↔ Response Alignment
- Extracts recommendations and surfaces likely responses/snippets.
- Uses lexical + heuristic alignment (with optional semantic search inputs).
- View and review alignments in the Alignment tab.

### 📁 Supported File Types

- **PDF** - Uses pdfplumber + PyPDF2 for text extraction
- **DOCX** - Microsoft Word documents via python-docx
- **TXT** - Plain text files with encoding detection

### 🔧 Advanced Features

- **Real-time Performance Tracking** - See search speed and efficiency
- **Search Analytics** - Track usage patterns and optimization
- **Content Analysis** - Automatic document feature detection
- **Smart Filtering** - Filter by file type, word count, etc.
- **Result Explanations** - Understand why results were matched
- **Search History** - Quick re-search previous queries

## Performance Benchmarks

| Search Mode | Speed | Accuracy | Best For |
|-------------|-------|----------|----------|
| Smart Pattern | ⚡ Very Fast | 📊 Good | Exact terms |
| RAG Semantic | 🤖 Medium | 🎯 Excellent | Concepts |
| Hybrid | 🔄 Medium | 🏆 Best | All cases |
| Fuzzy | 🎯 Fast | 📈 Good | Typos |

## Troubleshooting

### RAG Features Not Available
```
❌ RAG requires: pip install sentence-transformers torch scikit-learn
```

**Solution:**
```bash
pip install sentence-transformers torch scikit-learn
```

### PDF Processing Issues
```
❌ PDF processing not available
```

**Solution:**
```bash
pip install PyPDF2 pdfplumber
```

### Memory Issues with Large Documents
- Use fewer documents at once
- Try Smart Pattern search instead of RAG
- Close other applications to free memory

## Tips for Best Results

### 🎯 Search Query Tips
- **Specific terms**: Use exact terminology from your documents
- **Phrases**: Use quotes for exact phrases: `"policy implementation"`
- **Multiple concepts**: Combine related terms: `healthcare policy recommendations`
- **Context matters**: RAG search understands context better than keywords

### 📊 Performance Optimization
- **RAG indexing**: First search takes longer (building semantic index)
- **Subsequent searches**: Much faster once index is built
- **Document size**: Larger documents = slower RAG processing
- **Query length**: 2-8 words typically work best

### 🔍 Choosing Search Mode
- **Known terms** → Smart Pattern Search
- **Concepts/ideas** → RAG Semantic Search  
- **Best results** → Hybrid Search
- **Unsure spelling** → Fuzzy Search

## File Structure

```
DaphneAI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── config.toml                     # Streamlit config (if used)
├── daphne_core/                    # Core logic (search, extraction, alignment, utils)
├── ui/                             # UI modules (tabs, display helpers, search orchestration)
├── documentation/                  # Architecture and feature docs (new)
├── recsandresps/                   # Dataset/table-of-contents reference
├── samplepdfs/                     # Sample documents
├── PLAN.md                         # Work plan
├── licence.txt
└── README.md
```

## Next Steps

1. **Upload some documents** - PDF, DOCX, or TXT files
2. **Try different search modes** - Compare results between RAG and Smart search
3. **Use advanced options** - Try filters and different result counts
4. **Check analytics** - View performance metrics and search history

## Need Help?

- Check the error messages in the sidebar
- Try the sample data option if having upload issues
- Start with Smart Pattern search if RAG isn't working
- Ensure all dependencies are installed correctly

---

**🎉 You're ready to search!** Upload documents and start exploring the powerful search capabilities.
