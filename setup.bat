@echo off
REM Chat RAG AI Agent - Setup and Run Script (Windows)

echo.
echo 🚀 Chat RAG AI Agent - Setup Script
echo ====================================

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

echo ✅ Docker found

REM Check if docker-compose is installed
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed.
    exit /b 1
)

echo ✅ Docker Compose found

REM Create upload directory
if not exist "static\uploads" mkdir static\uploads
echo ✅ Created upload directory

REM Build and start services
echo.
echo 🐳 Building and starting Docker services...
docker-compose up -d

echo.
echo ⏳ Waiting for services to be healthy...
timeout /t 10

REM Check status
echo.
echo 🔍 Checking service status...
docker-compose ps

echo.
echo ✅ All services started successfully!
echo.
echo 📝 API Documentation:
echo    - Swagger UI: http://localhost:8000/docs
echo    - ReDoc: http://localhost:8000/redoc
echo.
echo 🏥 Health Check:
echo    - curl http://localhost:8000/health/
echo.
echo 📚 Database:
echo    - PostgreSQL: localhost:5432
echo    - Qdrant: localhost:6333
echo    - Ollama: localhost:11434
echo.
echo 💡 Next Steps:
echo    1. Download the Postman collection: Chat-RAG-AI-Agent.postman_collection.json
echo    2. Register a new user via /auth/register
echo    3. Login and get tokens via /auth/login
echo    4. Upload files via /files/upload
echo    5. Sync embeddings via /files/sync-embeddings/{file_id}
echo    6. Ask questions via /chat/ask
echo.
echo 🛑 To stop services: docker-compose down
