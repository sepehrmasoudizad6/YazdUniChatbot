# 🚀 Quick Start Guide

Get the Yazd University Intelligent Assistant running in 5 minutes!

## ⚡ Quick Setup

### 1. Prerequisites
- Python 3.9+ installed
- AVALAI API key (get from [avalai.ir](https://avalai.ir))

### 2. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file:
```bash
echo "AVALAI_API_KEY=your_api_key_here" > .env
```

### 4. Run the Application

```bash
# Use the startup script (recommended)
python run.py

# OR run manually:
python backend.py
```

### 5. Access the Application

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:8501 (if using `python run.py`)

## 🧪 Test the Application

### Test with curl:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "ایمیل استاد جهانگرد چیست؟"}'
```

### Test with Python:
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "دانشکده استاد محمدی"
})
print(response.json()["answer"])
```

## 🔧 Troubleshooting

### Common Issues:

1. **"AVALAI_API_KEY not found"**
   - Make sure your `.env` file contains the correct API key

2. **"Database not found"**
   - Run: `python create_database.py`

3. **"Port already in use"**
   - Kill existing processes or change port in `.env`

4. **"Virtual environment not activated"**
   - Run: `source .venv/bin/activate`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [API documentation](http://localhost:8000/docs)
- Check out the [usage examples](README.md#usage-examples)

---

**Need help?** Check the [troubleshooting section](README.md#troubleshooting) in the full README. 