# 📋 Recommendation-Response Tracker

An AI-powered document analysis system that extracts recommendations from documents, annotates them with conceptual themes using BERT, and finds corresponding responses using RAG (Retrieval-Augmented Generation) techniques.

## 🌟 Features

- **Document Processing**: Upload and extract text from PDF documents
- **AI-Powered Extraction**: Extract recommendations using GPT and pattern-based methods
- **BERT Annotation**: Annotate recommendations with conceptual frameworks (I-SIRch, House of Commons, custom)
- **Response Matching**: Find responses to recommendations using semantic search and concept matching
- **Interactive Dashboard**: Visualize analysis results and export data

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/recommendation-response-tracker.git
   cd recommendation-response-tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🏗️ Architecture

The system consists of several key components:

### Core Modules

- **`document_processor.py`**: PDF text extraction using pdfplumber and PyMuPDF
- **`llm_extractor.py`**: AI-powered recommendation extraction using OpenAI GPT
- **`bert_annotator.py`**: BERT-based concept annotation with multiple frameworks
- **`vector_store.py`**: Document indexing using ChromaDB and OpenAI embeddings
- **`rag_engine.py`**: Retrieval-Augmented Generation for finding responses
- **`recommendation_matcher.py`**: Multi-modal matching combining semantic similarity and concept overlap

### Annotation Frameworks

1. **I-SIRch Framework**
   - External - Policy factors
   - System - Organizational factors
   - Technology - Technology and tools
   - Person - Staff characteristics
   - Task - Task characteristics

2. **House of Commons Framework**
   - Communication
   - Fragmented care
   - Workforce pressures
   - Biases and stereotyping

3. **Extended Analysis Framework**
   - Procedural and Process Failures
   - Medication safety
   - Resource allocation

4. **Custom Frameworks**
   - Upload your own taxonomies in JSON/CSV/Excel format

## 📊 Usage Workflow

1. **Upload Documents**: Upload PDF files containing recommendations and responses
2. **Extract Recommendations**: Use AI or pattern-based methods to identify recommendations
3. **Annotate with Concepts**: Apply BERT-based annotation to identify themes
4. **Index Documents**: Create vector embeddings for semantic search
5. **Find Responses**: Match recommendations to responses using multi-modal approach
6. **Analyze Results**: View analytics dashboard and export data

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_STORE_PATH=./data/vector_store
LOG_LEVEL=INFO
```

### Model Configuration

The system uses:
- **BERT Model**: `emilyalsentzer/Bio_ClinicalBERT` (configurable)
- **LLM Model**: `gpt-3.5-turbo` (configurable)
- **Embeddings**: OpenAI text-embedding-ada-002

## 📁 Project Structure

```
recommendation-response-tracker/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── config.toml                    # Configuration file
├── .env.example                   # Environment variables template
└── modules/
    ├── __init__.py               # Package initialization
    ├── core_utils.py             # Core utilities and data classes
    ├── document_processor.py     # PDF processing
    ├── llm_extractor.py         # LLM-based extraction
    ├── bert_annotator.py        # BERT concept annotation
    ├── vector_store.py          # Vector database management
    ├── rag_engine.py            # RAG query processing
    ├── recommendation_matcher.py # Response matching
    └── streamlit_components.py   # UI components
```

## 🎯 Key Features Explained

### 1. Multi-Modal Recommendation Extraction

- **AI-Powered**: Uses OpenAI GPT for intelligent extraction
- **Pattern-Based**: Regex patterns for structured documents
- **Hybrid Approach**: Combines both methods for maximum coverage

### 2. BERT-Based Concept Annotation

- **Semantic Understanding**: Uses Clinical BERT for healthcare contexts
- **Multiple Frameworks**: Support for established and custom taxonomies
- **Confidence Scoring**: Provides reliability metrics for annotations

### 3. Advanced Response Matching

- **Semantic Similarity**: Vector-based document comparison
- **Concept Overlap**: Theme-based matching validation
- **Combined Confidence**: Multi-factor scoring system

### 4. Interactive Analytics

- **Real-time Processing**: Live updates as documents are processed
- **Visual Analytics**: Charts and metrics for pattern analysis
- **Export Capabilities**: CSV/JSON export for further analysis

## 🔬 Technical Details

### BERT Annotation Process

1. **Text Preprocessing**: Clean and normalize input text
2. **Embedding Generation**: Create BERT embeddings for documents and themes
3. **Similarity Calculation**: Compute cosine similarity between embeddings
4. **Keyword Matching**: Validate with keyword-based scoring
5. **Combined Scoring**: Merge semantic and keyword scores

### RAG Response Finding

1. **Document Indexing**: Create vector embeddings for all documents
2. **Query Processing**: Transform recommendation into search query
3. **Similarity Search**: Find semantically similar document chunks
4. **Response Filtering**: Identify response-type documents
5. **Ranking**: Score and rank potential matches

## 🚀 Deployment

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with debug mode
streamlit run app.py --logger.level=debug
```

### Production Deployment

```bash
# Use production requirements
pip install -r requirements.txt --no-dev

# Set production environment variables
export OPENAI_API_KEY=your_key
export LOG_LEVEL=WARNING

# Run with optimizations
streamlit run app.py --server.headless=true
```

## 🔒 Security Considerations

- API keys stored in environment variables
- File upload validation and sanitization
- Size limits on uploaded documents
- Input text cleaning and validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m "Add new feature"`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/recommendation-response-tracker/issues) page
2. Create a new issue with detailed information
3. Include error logs and steps to reproduce

## 🔮 Future Enhancements

- [ ] Support for additional document formats (Word, Excel)
- [ ] Advanced visualization with network graphs
- [ ] Integration with external databases
- [ ] Real-time collaboration features
- [ ] API endpoints for programmatic access
- [ ] Advanced ML models for domain-specific extraction

## 📊 Performance

The system is optimized for:
- **Document Processing**: ~10 pages per second
- **BERT Annotation**: ~50 recommendations per minute
- **Vector Search**: Sub-second response times
- **Memory Usage**: Efficient chunking and garbage collection

## 🏥 Healthcare Focus

Originally designed for healthcare recommendation tracking, the system supports:
- Clinical guidelines and recommendations
- Safety incident analysis
- Policy implementation tracking
- Quality improvement initiatives

---

Built with ❤️ using Streamlit, BERT, and OpenAI
