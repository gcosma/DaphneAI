# 📋 Recommendation-Response Tracker

An AI-powered document analysis system that extracts recommendations from UK Government inquiry reports, annotates them with conceptual themes using BERT, and finds corresponding responses using advanced AI techniques.

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## 🌟 Features

### 🔍 **Advanced Extraction**
- **Smart Complete Extraction** - Captures full recommendations, not fragments
- **AI-Powered Extraction** - Uses OpenAI GPT-3.5-turbo for intelligent analysis
- **Free AI Features** - BERT and semantic analysis without API costs
- **Multi-format Support** - PDF processing with multiple extraction methods

### 🤖 **AI Capabilities**
- **Clinical BERT** - Healthcare-specialized language understanding
- **Sentence Transformers** - Semantic similarity and clustering
- **Topic Modeling** - Automatic grouping of related recommendations
- **Sentiment Analysis** - Emotion and urgency detection
- **Quality Scoring** - AI-powered confidence and completeness metrics

### 📊 **Analysis & Visualization**
- **Interactive Dashboard** - Real-time analytics and visualizations
- **Response Matching** - Find government responses to recommendations
- **Export Capabilities** - CSV, JSON, and formatted reports
- **Batch Processing** - Handle multiple documents efficiently

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/recommendation-response-tracker.git
cd recommendation-response-tracker
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# For enhanced AI features (optional)
pip install sentence-transformers transformers torch scikit-learn
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (OpenAI API key is optional)
nano .env
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🏗️ Architecture

### Core Components

```
recommendation-response-tracker/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
└── modules/
    ├── __init__.py                 # Package initialization
    ├── ui/                         # User interface components
    │   ├── __init__.py
    │   ├── upload_components.py    # Document upload interface
    │   ├── extraction_components.py # AI extraction interface
    │   ├── annotation_components.py # Concept annotation
    │   ├── matching_components.py   # Response matching
    │   ├── search_components.py     # Smart search interface
    │   └── dashboard_components.py  # Analytics dashboard
    ├── llm_extractor.py            # OpenAI GPT integration
    ├── document_processor.py       # PDF text extraction
    └── core_utils.py               # Utility functions
```

## 📊 Usage Workflow

### 1. **Upload Documents** 📁
- Upload PDF files containing inquiry reports and government responses
- Automatic text extraction and document categorization
- Duplicate detection and validation
- Batch upload support

### 2. **Extract Content** 🔍
Choose from multiple extraction methods:
- **🧠 Smart Complete** - Context-aware multi-line extraction
- **🤖 AI-Powered** - OpenAI GPT analysis (requires API key)
- **🔬 BERT Analysis** - Free semantic understanding
- **🎯 Ensemble** - Combined approach for maximum accuracy

### 3. **Analyze Results** 📊
- Quality scoring and confidence metrics
- Topic clustering and theme identification
- Sentiment analysis and urgency detection
- Interactive previews and detailed analytics

### 4. **Export Data** 📥
- **CSV Export** - Comprehensive data with all metrics
- **JSON Export** - Complete metadata and analysis
- **Formatted Reports** - Human-readable summaries
- **BERT Analysis** - Specialized semantic data

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Optional: OpenAI API for enhanced features
OPENAI_API_KEY=your_openai_api_key_here

# Application settings
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=200
BATCH_SIZE=10

# AI model configuration
BERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
CONFIDENCE_THRESHOLD=0.7
```

### AI Models Used

| Model | Purpose | Size | Cost |
|-------|---------|------|------|
| **OpenAI GPT-3.5-turbo** | Intelligent extraction | API | ~$0.002/1K tokens |
| **Clinical BERT** | Healthcare context | ~400MB | Free |
| **all-MiniLM-L6-v2** | Semantic similarity | ~90MB | Free |
| **Smart Extraction** | Pattern-based | N/A | Free |

## 🎯 Key Features Explained

### Smart Complete Extraction
- **Multi-line capture** - Gets complete recommendations instead of fragments
- **Context-aware boundaries** - Knows when content starts and stops
- **Quality assessment** - Rates extraction completeness and accuracy

### AI-Powered Analysis
- **Semantic understanding** - Uses BERT for meaning-based analysis
- **Topic clustering** - Groups related recommendations automatically
- **Response classification** - Identifies accepted/rejected/partial responses
- **Confidence scoring** - Provides reliability metrics for all extractions

### Advanced Analytics
- **Quality distribution** - High/medium/low quality breakdowns
- **Content analysis** - Word counts, structure assessment
- **Method comparison** - Shows results from different extraction approaches
- **Export flexibility** - Multiple formats for different use cases

## 🔒 Security & Privacy

- **Local processing** - Most AI features run locally (no data sent to external services)
- **API key protection** - OpenAI keys stored securely in environment variables
- **File validation** - Input sanitization and size limits
- **Data isolation** - Each session maintains separate document storage

## 🚀 Deployment Options

### Local Development
```bash
# Run with debug mode
streamlit run app.py --logger.level=debug

# With custom port
streamlit run app.py --server.port=8502
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements.txt --no-dev

# Set production environment
export LOG_LEVEL=WARNING
export DEBUG_MODE=False

# Run with optimizations
streamlit run app.py --server.headless=true
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless=true"]
```

## 🧪 Testing

### Quick Test
1. Start the application: `streamlit run app.py`
2. Upload a sample PDF document
3. Try "Smart Complete Extraction" (works without API keys)
4. View results and download exports

### Full AI Test
1. Add OpenAI API key to `.env` file
2. Install AI dependencies: `pip install sentence-transformers scikit-learn`
3. Restart application
4. Try "AI-Powered Extraction" or "Full Ensemble"
5. Compare results across different methods

## 📈 Performance

### Extraction Speed
- **Smart Complete**: ~1-2 seconds per document
- **AI-Powered**: ~5-10 seconds per document (depends on API)
- **BERT Analysis**: ~3-5 seconds per document
- **Full Ensemble**: ~10-15 seconds per document

### Resource Usage
- **Memory**: ~500MB base + ~1GB for AI models
- **Storage**: ~1.5GB for all AI models (one-time download)
- **CPU**: Moderate usage, GPU optional for BERT

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m "Add new feature"`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/recommendation-response-tracker.git

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### Getting Help
1. Check the [Issues](https://github.com/yourusername/recommendation-response-tracker/issues) page
2. Review the documentation and README
3. Create a new issue with detailed information

### Common Issues
- **PDF extraction fails**: Install `pip install pdfplumber PyMuPDF`
- **AI features unavailable**: Install `pip install sentence-transformers transformers torch`
- **High memory usage**: Reduce batch size or disable AI features
- **Slow performance**: Use "Smart Complete" extraction only

## 🔄 Changelog

### Version 2.0.0
- ✅ Complete AI integration with multiple models
- ✅ Smart complete extraction for better accuracy
- ✅ Enhanced UI with comprehensive error handling
- ✅ Free AI features requiring no API keys
- ✅ Advanced export capabilities with multiple formats

### Version 1.0.0
- ✅ Basic PDF upload and text extraction
- ✅ Pattern-based recommendation extraction
- ✅ Simple export functionality

## 🎖️ Acknowledgments

- **Streamlit** for the excellent web app framework
- **OpenAI** for GPT language models
- **Hugging Face** for transformer models and infrastructure
- **UK Government** for open access to inquiry reports and responses

---

**Built with ❤️ for analyzing UK Government inquiry reports and responses**
