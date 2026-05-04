# Project File Structure and Description

## Complete Directory Tree

```
Chat-RAG-AI-Agent/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   │
│   ├── api/                      # API route handlers
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication endpoints
│   │   ├── files.py             # File upload endpoints
│   │   ├── chat.py              # Chat/RAG endpoints
│   │   ├── health.py            # Health check endpoints
│   │   └── dependencies.py      # JWT dependency injection
│   │
│   ├── core/                     # Core application configuration
│   │   ├── __init__.py
│   │   ├── config.py            # Settings management (Pydantic)
│   │   ├── security.py          # JWT token handling & bcrypt
│   │   ├── logging.py           # Logging configuration
│   │   └── middleware.py        # Custom middleware
│   │
│   ├── models/                   # SQLAlchemy ORM models
│   │   └── __init__.py          # User, File, ChatHistory models
│   │
│   ├── schemas/                  # Pydantic request/response schemas
│   │   └── __init__.py          # All API schemas
│   │
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication service
│   │   ├── file.py              # File management service
│   │   └── chat.py              # Chat service with RAG
│   │
│   ├── repositories/             # Data access layer (CRUD)
│   │   ├── __init__.py
│   │   ├── base.py              # Base repository class
│   │   ├── user.py              # User repository
│   │   ├── file.py              # File repository
│   │   └── chat.py              # Chat history repository
│   │
│   ├── ai/                       # AI/ML components
│   │   ├── __init__.py
│   │   ├── llm.py               # Ollama LLM client
│   │   ├── vector_db.py         # Qdrant vector database client
│   │   ├── rag.py               # RAG pipeline orchestration
│   │   ├── agent.py             # ReAct agent logic
│   │   └── text_processor.py    # Text chunking & preprocessing
│   │
│   ├── db/                       # Database setup & management
│   │   ├── __init__.py
│   │   └── database.py          # SQLAlchemy async setup
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── helpers.py           # Helper utilities
│       └── exceptions.py        # Custom exception classes
│
├── static/
│   └── uploads/                 # File upload directory
│       └── .gitkeep
│
├── alembic/                      # Database migration management
│   ├── env.py                   # Alembic environment configuration
│   ├── alembic.ini              # Alembic configuration
│   └── versions/
│       └── 001_initial.py       # Initial migration
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   └── test_auth.py            # Authentication tests
│
├── .env                         # Environment variables (sample)
├── .env.prod                    # Production environment variables
├── .gitignore                   # Git ignore patterns
├── .dockerignore                # Docker ignore patterns
│
├── Dockerfile                   # Docker container image
├── docker-compose.yml           # Docker Compose orchestration
│
├── requirements.txt             # Python dependencies
├── conftest.py                  # Pytest configuration
│
├── setup.sh                     # Setup script (Linux/Mac)
├── setup.bat                    # Setup script (Windows)
├── test_api.sh                  # API testing script
├── example_usage.py             # Python client example
│
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── DEPLOYMENT.md               # Deployment guidelines
├── PROJECT_STRUCTURE.md        # This file
│
└── Chat-RAG-AI-Agent.postman_collection.json  # Postman collection
```

## File Descriptions

### Core Application Files

#### `app/main.py`
- FastAPI application factory
- Lifespan event handlers for startup/shutdown
- Route registration
- Middleware configuration
- CORS setup

#### `app/core/config.py`
- Settings management using Pydantic v2
- Environment variable loading
- Type-safe configuration access
- Cached settings instance

#### `app/core/security.py`
- JWT token creation and validation
- Password hashing with bcrypt
- Token encoding/decoding
- Access and refresh token generation

#### `app/core/logging.py`
- Logging configuration
- File and console handlers
- Structured logging setup

#### `app/core/middleware.py`
- Error handling middleware
- Request/response logging middleware
- Custom middleware utilities

### Database Layer

#### `app/models/__init__.py`
- User model (id, email, password_hash, timestamps)
- File model (id, filename, filepath, metadata)
- ChatHistory model (id, question, answer, sources)
- SQLAlchemy declarative base

#### `app/db/database.py`
- Async PostgreSQL engine creation
- AsyncSession factory
- Connection pooling configuration
- Database initialization functions

#### `app/repositories/*.py`
- Base repository with CRUD operations
- Specialized repositories for User, File, ChatHistory
- Query methods for common operations

### Service Layer

#### `app/services/auth.py`
- User registration with validation
- Login with password verification
- Token generation
- User retrieval operations

#### `app/services/file.py`
- File upload handling
- File metadata management
- File deletion (soft delete)
- File content reading
- Embedding synchronization

#### `app/services/chat.py`
- Question processing with RAG
- ReAct agent orchestration
- Chat history storage
- Source reference extraction

### API Layer

#### `app/api/auth.py`
- POST /auth/register - User registration
- POST /auth/login - User login
- GET /auth/me - Get current user

#### `app/api/files.py`
- POST /files/upload - Upload file
- GET /files/list - List user files
- DELETE /files/{file_id} - Delete file
- POST /files/sync-embeddings/{file_id} - Sync embeddings

#### `app/api/chat.py`
- POST /chat/ask - Ask question with RAG
- GET /chat/history - Get chat history

#### `app/api/health.py`
- GET /health/ - Quick health check
- GET /health/detailed - Detailed health check

#### `app/api/dependencies.py`
- JWT dependency injection
- User authentication
- Current user extraction from token

### AI/ML Components

#### `app/ai/llm.py`
- Ollama LLM client
- Text generation
- Embedding generation
- Health check

#### `app/ai/vector_db.py`
- Qdrant vector database client
- Vector upsert operations
- Similarity search
- Collection initialization

#### `app/ai/rag.py`
- RAG pipeline orchestration
- Document processing
- Vector search and retrieval
- Pipeline health checks

#### `app/ai/agent.py`
- ReAct (Reason + Act) agent
- Thought generation
- Action planning
- Response generation
- Multi-step reasoning

#### `app/ai/text_processor.py`
- Text chunking with overlap
- Text cleaning and normalization
- PDF text extraction
- Sentence-based chunking

### Schemas

#### `app/schemas/__init__.py`
- UserRegisterRequest/Response
- UserLoginRequest
- TokenResponse
- FileUploadResponse, FileListResponse
- ChatRequest, ChatResponse
- ChatHistoryResponse
- HealthCheckResponse

### Utilities

#### `app/utils/helpers.py`
- UUID generation and validation
- Token extraction from JWT
- Email validation
- File size formatting

#### `app/utils/exceptions.py`
- Custom HTTP exceptions
- InvalidCredentialsException
- UnauthorizedException
- ForbiddenException
- ValidationException

### Configuration Files

#### `.env`
- Environment variables for local development
- Database, Vector DB, LLM URLs
- JWT secrets and expiry
- File upload settings

#### `docker-compose.yml`
- PostgreSQL service
- Qdrant service
- Ollama service
- FastAPI service
- Redis service (optional)
- Network configuration
- Volume management
- Health checks

#### `Dockerfile`
- Multi-stage build for optimization
- Python 3.11 slim base image
- Dependency installation
- Application deployment
- Health check configuration

#### `requirements.txt`
- FastAPI and Uvicorn
- SQLAlchemy with asyncpg
- Pydantic v2
- JWT libraries (PyJWT)
- bcrypt for password hashing
- Qdrant client
- LangChain
- Testing libraries (pytest)

### Testing

#### `conftest.py`
- Pytest fixtures
- Async engine and session fixtures
- Event loop management

#### `tests/test_auth.py`
- User registration tests
- Login tests
- Password validation tests
- Error handling tests

### Documentation

#### `README.md`
- Complete project documentation
- Architecture overview
- Installation instructions
- API usage examples
- Deployment guidelines
- Troubleshooting guide

#### `QUICKSTART.md`
- 5-minute setup guide
- Quick API testing
- Docker commands
- First API calls

#### `DEPLOYMENT.md`
- Deployment strategies
- AWS ECS deployment
- Kubernetes deployment
- Production checklist
- CI/CD pipeline setup
- Monitoring setup

#### `PROJECT_STRUCTURE.md`
- This file
- Complete file tree
- File descriptions

### Helper Scripts

#### `setup.sh` (Linux/Mac)
- Automated setup script
- Docker service startup
- Health verification

#### `setup.bat` (Windows)
- Windows setup script
- Docker service startup
- Service verification

#### `test_api.sh`
- API endpoint testing
- Health checks
- Sample requests

#### `example_usage.py`
- Python client implementation
- Async API usage examples
- Full workflow demonstration

#### `Chat-RAG-AI-Agent.postman_collection.json`
- Postman collection
- All API endpoints
- Example requests
- Variable management

## Data Flow

### Authentication Flow
1. User registers via `/auth/register`
2. Password hashed with bcrypt, stored in PostgreSQL
3. User logs in via `/auth/login`
4. JWT tokens generated (access + refresh)
5. Token included in subsequent requests via Authorization header

### File Upload & RAG Flow
1. User uploads file via `/files/upload`
2. File saved to `static/uploads/`
3. Metadata stored in PostgreSQL
4. User calls `/files/sync-embeddings/{file_id}`
5. File content extracted and chunked
6. Chunks embedded using Ollama
7. Vectors stored in Qdrant with metadata

### Query & Response Flow
1. User asks question via `/chat/ask`
2. Question embedded using Ollama
3. Semantic search in Qdrant retrieves relevant chunks
4. ReAct agent processes question + context
5. Ollama generates response
6. Response stored in ChatHistory
7. Sources and answer returned to user

## Technology Stack Summary

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Async Runtime | asyncio |
| Database | PostgreSQL + SQLAlchemy async |
| Vector DB | Qdrant |
| LLM | Ollama |
| Auth | JWT + bcrypt |
| Validation | Pydantic v2 |
| Container | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |
| AI/ML | LangChain + LangGraph |

---

**Last Updated**: January 2024
**Version**: 1.0.0
