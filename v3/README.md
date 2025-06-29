# 🎓 Yazd University Intelligent Assistant

A comprehensive AI-powered assistant for Yazd University that provides intelligent responses to queries about professors, courses, and academic information using Persian language processing and advanced vector search capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Persian Language Processing**: Specialized embeddings for Persian text
- **Intelligent Query Processing**: Advanced query reformatting and enhancement
- **Context-Aware Retrieval**: Reranking and compression for better results
- **Metadata-Based Search**: Search by professor names, faculties, addresses
- **Self-Query Retriever**: Structured searches using metadata fields
- **Comprehensive Logging**: Detailed logging and monitoring
- **RESTful API**: FastAPI-based backend with automatic documentation
- **Error Handling**: Robust error handling and validation

## 🔧 Prerequisites

Before running this project, ensure you have:

- **Python 3.9+** installed
- **Git** for cloning the repository
- **AVALAI API Key** for the language model (get from [avalai.ir](https://avalai.ir))
- **Sufficient disk space** for the vector database (~100MB)
- **Internet connection** for downloading AI models

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd v3
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
langchain==0.0.350
langchain-openai==0.0.2
langchain-huggingface==0.0.6
langchain-chroma==0.0.1
langchain-community==0.0.10
pydantic==2.5.0
torch==2.1.1
chromadb==0.4.18
sentence-transformers==2.2.2
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# API Configuration
AVALAI_API_KEY=your_avalai_api_key_here

# Application Configuration
DEBUG=False
HOST=0.0.0.0
PORT=8000

# Model Configuration
EMBEDDINGS_MODEL=heydariAI/persian-embeddings
LLM_MODEL=gpt-4.1
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
```

## 🗄️ Database Setup

### Option 1: Use Existing Database (Recommended)

The project already includes a pre-built database in the `chroma_db/` directory. If you want to use it:

```bash
# The database is already set up and ready to use
# No additional steps required
```

### Option 2: Rebuild Database

If you want to rebuild the database from scratch:

```bash
# Run the database creation script
python create_database.py
```

This will:
- Read documents from `scraped_content/` directory
- Extract metadata (نام, دانشکده, آدرس بخش, آدرس شخصی)
- Create embeddings using Persian language model
- Store in `chroma_db/` directory

## 🚀 Running the Application

### 1. Start the Backend Server

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Start the FastAPI server
python backend.py
```

The server will start on `http://localhost:8000`

### 2. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

### 3. Start the Frontend (Optional)

```bash
# In a new terminal
python frontend.py
```

The frontend will be available at `http://localhost:8501`

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with basic info |
| `/health` | GET | Health check |
| `/metrics` | GET | Application metrics |
| `/query` | POST | Process user queries |
| `/info` | GET | System information |

### Database Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/database/info` | GET | Database information |
| `/database/metadata/fields` | GET | Available metadata fields |
| `/search/metadata` | POST | Search by metadata |

## 📝 Usage Examples

### 1. Basic Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ایمیل استاد جهانگرد چیست؟",
    "user_id": "user_123"
  }'
```

### 2. Metadata Search

```bash
curl -X POST "http://localhost:8000/search/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "field": "نام",
    "value": "احمدی",
    "limit": 5
  }'
```

### 3. Get Database Info

```bash
curl "http://localhost:8000/database/info"
```

### 4. Python Client Example

```python
import requests

# Query the assistant
response = requests.post("http://localhost:8000/query", json={
    "query": "دانشکده استاد محمدی",
    "user_id": "user_123"
})

print(response.json()["answer"])
```

## 🔍 Available Metadata Fields

The database supports the following metadata fields for searching:

- **نام**: Professor name
- **دانشکده**: Faculty/College
- **آدرس بخش**: Department address
- **آدرس شخصی**: Personal address
- **file_id**: Original file identifier

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: AVALAI_API_KEY not found
   ```
   **Solution**: Ensure your `.env` file contains the correct API key

2. **Model Download Issues**
   ```
   Error: Failed to load embeddings model
   ```
   **Solution**: Check internet connection and try again

3. **Database Not Found**
   ```
   Error: Database path not found
   ```
   **Solution**: Run `python create_database.py` to rebuild the database

4. **Port Already in Use**
   ```
   Error: Address already in use
   ```
   **Solution**: Change the port in `.env` file or kill the existing process

### Performance Optimization

- **GPU Acceleration**: The system automatically uses CUDA if available
- **Memory Usage**: Monitor memory usage, especially with large databases
- **Caching**: Results are cached for better performance

### Logs

- **Application Logs**: `logs/app_YYYYMMDD.log`
- **Error Logs**: `logs/errors_YYYYMMDD.log`
- **Database Logs**: `database_creation.log`

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Database Statistics

```bash
curl http://localhost:8000/database/info
```

## 🔒 Security Considerations

- Keep your API key secure and never commit it to version control
- Use environment variables for sensitive configuration
- Consider implementing authentication for production use
- Monitor API usage and implement rate limiting if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the logs for error details
- Open an issue on the repository

---

**Version**: 2.1.0  
**Last Updated**: 2024  
**Author**: AI Assistant 