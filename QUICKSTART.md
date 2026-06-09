# Quick Start Guide for Chat RAG AI Agent

## 🚀 Five-Minute Setup

### Option 1: Docker (Recommended)

1. **Clone/Download the project**
   ```bash
   cd Chat-RAG-AI-Agent
   ```

2. **Start all services**
   ```bash
   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   
   # Windows
   setup.bat
   ```

3. **Verify services are running**
   ```bash
   curl http://localhost:8000/health/
   ```

4. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Option 2: Manual Setup (Local Python)

1. **Install Python 3.11+**
   ```bash
   python --version  # Should be 3.11+
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start required services**
   - PostgreSQL on port 5432
   - Qdrant on port 6333
   - Ollama on port 11434

   (Or use docker for individual services)

5. **Update .env if needed**
   ```bash
   cp .env .env.local
   # Edit connection strings if services are on different hosts
   ```

6. **Start the API**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📝 First API Calls

### 1. Register a User
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

**Save the response - you need the `id` for later.**

### 2. Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

**Save the `access_token` - use it in Authorization header.**

### 3. Check Health
```bash
curl "http://localhost:8000/health/"
```

Should return:
```json
{
  "status": "healthy",
  "database": "healthy",
  "vector_db": "healthy",
  "llm": "healthy"
}
```

### 4. Upload a File
```bash
# Create a test file
echo "Artificial Intelligence is transforming the world." > test.txt

# Upload it
curl -X POST "http://localhost:8000/files/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@test.txt"
```

**Save the `id` from the response - you'll need it next.**

### 5. Sync Embeddings (Generate Vector Embeddings)
```bash
curl -X POST "http://localhost:8000/files/sync-embeddings/FILE_ID" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

This creates embeddings and stores them in Qdrant.

### 6. Ask a Question (RAG + Agent)
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is transforming the world?"}'
```

You should get a response with:
- **answer**: The AI-generated answer
- **sources**: References to the document chunks used

## 📊 Using Postman

1. Download the Postman collection:
   - File: `Chat-RAG-AI-Agent.postman_collection.json`

2. Import into Postman:
   - Menu → Import → Select the JSON file

3. Set environment variables:
   - Create environment with `base_url` = `http://localhost:8000`
   - After login, set `access_token` variable

4. Run requests in sequence:
   - Register → Login → Upload → Sync Embeddings → Ask Question

## 🐳 Docker Commands Reference

```bash
# Add models to ollama for embadding and chat, This will be in volumnes to need to get again after cocker compose down -v
docker exec -it chat-rag-ollama ollama pull bge-m3
docker exec -it chat-rag-ollama ollama pull qwen3:8b

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs for API
docker-compose logs -f api

# View logs for Ollama
docker-compose logs -f ollama

# Access database shell
docker-compose exec postgres psql -U postgres -d chat_rag_db

# Access API container shell
docker-compose exec api bash

# Restart a specific service
docker-compose restart api

# Remove all data (containers + volumes)
docker-compose down -v
```

## 🔍 Troubleshooting

### Services won't start
```bash
# Check if ports are already in use
lsof -i :8000   # API
lsof -i :5432   # PostgreSQL
lsof -i :6333   # Qdrant
lsof -i :11434  # Ollama
```

### Database connection error
```bash
# Check PostgreSQL is running and accessible
docker-compose logs postgres

# Verify database exists
docker-compose exec postgres psql -U postgres -l
```

### Ollama model issues
```bash
# Check available models
curl http://localhost:11434/api/tags

# Pull the qwen3:8b model (if not present)
docker-compose exec ollama ollama pull qwen3:8b
docker-compose exec ollama ollama pull bge-m3
```

### Qdrant collection error
```bash
# Access Qdrant web UI
# http://localhost:6333/dashboard

# Clear data if needed
docker-compose down -v
docker-compose up -d
```

## 📚 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_auth.py

# Run with coverage
pytest --cov=app
```

## 🧬 Using Python Client

```python
from example_usage import ChatRAGClient
import asyncio

async def main():
    client = ChatRAGClient()
    
    # Register
    await client.register("user@test.com", "password123")
    
    # Login
    await client.login("user@test.com", "password123")
    
    # Upload file
    result = await client.upload_file("mydocument.txt")
    file_id = result["id"]
    
    # Sync embeddings
    await client.sync_embeddings(file_id)
    
    # Ask question
    answer = await client.ask_question("What is in the file?")
    print(answer)
    
    await client.close()

asyncio.run(main())
```

## 🔑 Environment Variables

Key variables in `.env`:

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/chat_rag_db
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_EMBEDDINGS_PATH=/api/embeddings
SECRET_KEY=your-secret-key-min-32-chars
```

## 📈 Next Steps

1. **Add more documents**: Upload multiple files and ask cross-document questions
2. **Customize prompts**: Modify agent prompts in `app/ai/agent.py`
3. **Add tools**: Extend the agent with custom tools (calculator, search, etc.)
4. **Set up monitoring**: Add Sentry, DataDog, or similar
5. **Deploy**: Use Kubernetes, AWS ECS, or similar for production

## 💡 Pro Tips

- **Use Swagger UI** for interactive API testing: http://localhost:8000/docs
- **Monitor Ollama** performance and adjust chunk sizes in `.env`
- **Use Redis** for caching by uncommenting in docker-compose.yml
- **Set DEBUG=False** and use stronger `SECRET_KEY` for production
- **Enable HTTPS** and set proper CORS origins for production

## 🆘 Need Help?

- Check logs: `docker-compose logs api`
- Review README.md for detailed documentation
- Check Swagger/OpenAPI docs at http://localhost:8000/docs
- Review example_usage.py for Python client examples

---

**Happy RAG'ing! 🚀**
