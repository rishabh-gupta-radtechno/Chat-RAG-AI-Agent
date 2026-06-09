# Chat RAG AI Agent - Backend System

A production-grade FastAPI backend system for Retrieval-Augmented Generation (RAG) with JWT authentication, PostgreSQL, Qdrant vector database, and Ollama LLM integration.

## 🚀 Features

- **Authentication**: JWT-based user authentication with access and refresh tokens
- **File Management**: Upload and manage PDF, TXT, and DOCX files
- **RAG Pipeline**: Automatic document chunking, embedding generation, and semantic search
- **ReAct Agent**: Reason + Act pattern for intelligent question answering
- **Vector Database**: Qdrant for efficient vector similarity search
- **LLM Integration**: Ollama for local model inference
- **Async Operations**: Full async/await support with FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Logging**: Comprehensive logging throughout the application
- **Docker**: Complete Docker and docker-compose setup

## 📋 Architecture

```
app/
├── api/                    # API routes/endpoints
│   ├── auth.py            # Authentication endpoints
│   ├── files.py           # File upload endpoints
│   ├── chat.py            # Chat/RAG endpoints
│   ├── health.py          # Health check endpoints
│   └── dependencies.py    # JWT dependency injection
├── core/                  # Core configuration
│   ├── config.py          # Settings management
│   ├── security.py        # JWT and password handling
│   ├── logging.py         # Logging configuration
│   └── middleware.py      # Custom middleware
├── models/                # SQLAlchemy models
│   └── __init__.py        # User, File, ChatHistory models
├── schemas/               # Pydantic schemas
│   └── __init__.py        # Request/response schemas
├── services/              # Business logic
│   ├── auth.py           # Authentication service
│   ├── file.py           # File management service
│   └── chat.py           # Chat service with RAG
├── repositories/          # Data access layer
│   ├── base.py           # Base repository class
│   ├── user.py           # User repository
│   ├── file.py           # File repository
│   └── chat.py           # Chat history repository
├── ai/                    # AI/ML components
│   ├── llm.py            # Ollama LLM client
│   ├── vector_db.py      # Qdrant client
│   ├── rag.py            # RAG pipeline
│   ├── agent.py          # ReAct agent
│   └── text_processor.py # Text processing utilities
├── db/                    # Database setup
│   └── database.py       # SQLAlchemy async setup
├── utils/                # Utilities
│   ├── helpers.py        # Helper functions
│   └── exceptions.py     # Custom exceptions
└── main.py              # FastAPI application

static/uploads/          # File upload directory
alembic/                # Database migrations (optional)
```

## 🔧 Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Async**: asyncio with asyncpg
- **Database**: PostgreSQL with SQLAlchemy
- **Vector DB**: Qdrant
- **LLM**: Ollama (qwen3:8b)
- **Auth**: JWT (PyJWT + bcrypt)
- **Validation**: Pydantic v2
- **Container**: Docker & Docker Compose

## 📦 Installation

### Prerequisites

- Python 3.11+ or Docker
- PostgreSQL (if running locally)
- Qdrant (if running locally)
- Ollama (if running locally)

### Local Setup (Without Docker)

1. Clone the repository:
```bash
cd Chat-RAG-AI-Agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env .env.local
# Edit .env.local with your configuration
```

5. Ensure PostgreSQL, Qdrant, and Ollama are running, then start the app:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


### Docker Setup (Recommended)

1. Build and start all services:
```bash
docker-compose up -d
```

2. Check logs:
```bash
docker-compose logs -f api
```

3. Access API:
```
http://localhost:8000
```

### Add models to ollama for embadding and chat, This will be in volumnes to need to get again after- docker compose down -v , and- docker-compose up -d
1. Pull embadding model to generate embadding
```bash
docker exec -it chat-rag-ollama ollama pull bge-m3
```

2. Pull chat model for communication
```bash
docker exec -it chat-rag-ollama ollama pull bge-m3
docker exec -it chat-rag-ollama ollama pull qwen3:8b
```


## 🚀 Getting Started

### 1. Register a User

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 2. Login and Get Tokens

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 3. Upload a File

```bash
curl -X POST "http://localhost:8000/files/upload" \
  -H "Authorization: Bearer {access_token}" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "filename": "document.pdf",
  "filepath": "static/uploads/20240115_103000_document.pdf",
  "file_size": 1024000,
  "file_type": "pdf",
  "is_active": true,
  "is_embedded": false,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 4. Sync Embeddings for File

```bash
curl -X POST "http://localhost:8000/files/sync-embeddings/{file_id}" \
  -H "Authorization: Bearer {access_token}"
```

Response:
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440001",
  "chunks_created": 42,
  "status": "completed"
}
```

### 5. Ask a Question (RAG + Agent)

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Authorization: Bearer {access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings in the document?"
  }'
```

Response:
```json
{
  "answer": "Based on the document, the key findings are...",
  "sources": [
    {
      "filename": "document.pdf",
      "file_id": "550e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 5,
      "relevance_score": 0.89
    },
    {
      "filename": "document.pdf",
      "file_id": "550e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 12,
      "relevance_score": 0.87
    }
  ],
  "model": "bge-m3",
  "thinking": "Iterations: 1\nThought: The question asks for key findings..."
}
```

### 6. Get Chat History

```bash
curl -X GET "http://localhost:8000/chat/history?skip=0&limit=50" \
  -H "Authorization: Bearer {access_token}"
```

Response:
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "question": "What are the key findings in the document?",
    "answer": "Based on the document, the key findings are...",
    "sources": [...],
    "created_at": "2024-01-15T10:35:00Z"
  }
]
```

### 7. List User Files

```bash
curl -X GET "http://localhost:8000/files/list" \
  -H "Authorization: Bearer {access_token}"
```

Response:
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "filename": "document.pdf",
    "file_size": 1024000,
    "file_type": "pdf",
    "is_embedded": true,
    "created_at": "2024-01-15T10:30:00Z"
  }
]
```

### 8. Health Check

```bash
curl -X GET "http://localhost:8000/health/"
```

Response:
```json
{
  "status": "healthy",
  "database": "healthy",
  "vector_db": "healthy",
  "llm": "healthy"
}
```

## 📝 API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Authentication

The API uses JWT-based authentication:

1. **Register**: Create a new user account
2. **Login**: Receive access and refresh tokens
3. **Access Token**: Valid for 30 minutes, used for API requests
4. **Refresh Token**: Valid for 7 days, used to get new access token

Include the access token in the `Authorization` header:
```
Authorization: Bearer {access_token}
```

## 📁 File Upload

- **Supported Formats**: PDF, TXT, DOCX
- **Max File Size**: 50MB
- **Upload Directory**: `static/uploads/`

### File Processing Flow

1. Upload file → Stored in database and filesystem
2. Sync embeddings → Read file, chunk text, generate embeddings
3. Store vectors → Vectors stored in Qdrant with metadata
4. Query → User asks question → Vectors retrieved from Qdrant
5. Response → ReAct agent processes context and generates answer

## 🤖 RAG Pipeline

The RAG pipeline consists of:

1. **Chunking**: Text split into overlapping chunks (default: 1024 tokens, 128 overlap)
2. **Embedding**: Each chunk embedded using Ollama (768 dimensions)
3. **Storage**: Vectors stored in Qdrant with metadata (filename, page, chunk text)
4. **Retrieval**: Query embedded and compared against stored vectors
5. **Ranking**: Top-K similar documents returned (default: 5)

## 🧠 ReAct Agent

The agent follows the Reason + Act pattern:

1. **Thought**: Analyze question and context
2. **Action**: Decide on strategy (retrieve documents, search, etc.)
3. **Observation**: Process results from action
4. **Respond**: Generate final answer with citations

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE,
  updated_at TIMESTAMP WITH TIME ZONE
);
```

### Files Table
```sql
CREATE TABLE files (
  id UUID PRIMARY KEY,
  filename VARCHAR(255) NOT NULL,
  filepath VARCHAR(512) NOT NULL,
  file_size INTEGER,
  file_type VARCHAR(50),
  is_active BOOLEAN DEFAULT true,
  is_embedded BOOLEAN DEFAULT false,
  uploaded_by UUID FOREIGN KEY,
  created_at TIMESTAMP WITH TIME ZONE,
  updated_at TIMESTAMP WITH TIME ZONE
);
```

### Chat Histories Table
```sql
CREATE TABLE chat_histories (
  id UUID PRIMARY KEY,
  user_id UUID FOREIGN KEY,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  sources TEXT,
  model VARCHAR(100),
  created_at TIMESTAMP WITH TIME ZONE
);
```

## 🐳 Docker Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f api
```

### Rebuild Images
```bash
docker-compose up -d --build
```

### Access Container Shell
```bash
docker-compose exec api bash
```

### Remove All Data (Including Volumes)
```bash
docker-compose down -v
```

## ⚙️ Configuration

Environment variables (in `.env`):

```
# API
ENV=development
DEBUG=True

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/chat_rag_db

# JWT
SECRET_KEY=your-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Vector DB
QDRANT_URL=http://localhost:6333

# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_EMBEDDINGS_PATH=/api/embeddings

# File Upload
MAX_UPLOAD_SIZE=52428800  # 50MB

# Embedding
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
EMBEDDING_DIMENSION=768

# RAG
VECTOR_SEARCH_TOP_K=5
SIMILARITY_THRESHOLD=0.5

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
```

## 🧪 Testing

Run unit tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app
```

## 📊 Performance Considerations

- **Connection Pooling**: PostgreSQL connection pool size: 20, max overflow: 10
- **Vector Search**: Top-K retrieval (default: 5 documents)
- **Chunking**: Overlapping chunks for better context (128 token overlap)
- **Caching**: Redis support (optional)
- **Async Operations**: All endpoints are fully async

## 🔍 Logging

Logs are written to:
- **Console**: Real-time output
- **File**: `app.log` for persistent logging

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## 🚀 Production Deployment

### Pre-Production Checklist

1. **Update `.env` with production values**:
   - Set `ENV=production`
   - Set `DEBUG=False`
   - Update `SECRET_KEY` to a strong random value
   - Update database credentials
   - Configure CORS origins

2. **Database**:
   - Use managed PostgreSQL (AWS RDS, Azure Database, etc.)
   - Enable backups and point-in-time recovery
   - Use separate connection pooling

3. **Vector Database**:
   - Use managed Qdrant (Qdrant Cloud)
   - Enable persistence and backups

4. **LLM**:
   - Ensure Ollama has adequate resources
   - Monitor inference latency
   - Consider load balancing for multiple replicas

5. **Security**:
   - Use HTTPS/TLS for all connections
   - Implement rate limiting
   - Add request validation and sanitization
   - Use secrets management (AWS Secrets Manager, etc.)

6. **Monitoring**:
   - Set up application monitoring (Sentry, DataDog, etc.)
   - Monitor database performance
   - Track API response times
   - Set up alerts for errors and performance issues

7. **Deployment**:
   - Use container orchestration (Kubernetes, ECS, etc.)
   - Set up CI/CD pipeline
   - Use automated testing
   - Plan for scaling

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For issues, questions, or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- FastAPI for the excellent async web framework
- Qdrant for vector similarity search
- Ollama for local LLM inference
- LangChain for RAG and agent orchestration

---

**Happy RAG! 🚀**
