#!/bin/bash

# Chat RAG AI Agent - Setup and Run Script

set -e

echo "🚀 Chat RAG AI Agent - Setup Script"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker found"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker Compose found"

# Create upload directory
mkdir -p static/uploads
echo "✅ Created upload directory"

# Build and start services
echo ""
echo "🐳 Building and starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check if services are running
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📝 API Documentation:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo ""
echo "🏥 Health Check:"
echo "   - curl http://localhost:8000/health/"
echo ""
echo "📚 Database:"
echo "   - PostgreSQL: localhost:5432"
echo "   - Qdrant: localhost:6333"
echo "   - Ollama: localhost:11434"
echo ""
echo "💡 Next Steps:"
echo "   1. Download the Postman collection: Chat-RAG-AI-Agent.postman_collection.json"
echo "   2. Register a new user via /auth/register"
echo "   3. Login and get tokens via /auth/login"
echo "   4. Upload files via /files/upload"
echo "   5. Sync embeddings via /files/sync-embeddings/{file_id}"
echo "   6. Ask questions via /chat/ask"
echo ""
echo "🛑 To stop services: docker-compose down"
