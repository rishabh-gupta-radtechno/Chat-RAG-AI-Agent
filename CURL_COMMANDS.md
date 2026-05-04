#!/bin/bash

# Chat RAG AI Agent - Complete API Curl Commands Reference
# Usage: Copy and paste these commands to test the API

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL="http://localhost:8000"
USER_EMAIL="testuser@example.com"
USER_PASSWORD="testpassword123"

# Store tokens after login (these will be updated automatically in bash)
ACCESS_TOKEN=""
FILE_ID=""

echo "Chat RAG AI Agent - API Command Reference"
echo "=========================================="
echo ""
echo "Base URL: $BASE_URL"
echo ""

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

echo "🏥 HEALTH CHECK ENDPOINTS"
echo "========================="
echo ""

echo "1. Quick Health Check"
echo "curl -X GET '$BASE_URL/health/'"
echo ""

echo "2. Detailed Health Check"
echo "curl -X GET '$BASE_URL/health/detailed'"
echo ""

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

echo ""
echo "🔐 AUTHENTICATION ENDPOINTS"
echo "============================"
echo ""

echo "3. Register User"
echo "curl -X POST '$BASE_URL/auth/register' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{'"
echo "    \"email\": \"$USER_EMAIL\","
echo "    \"password\": \"$USER_PASSWORD\""
echo "  }'"
echo ""

echo "4. Login User (Get Tokens)"
echo "curl -X POST '$BASE_URL/auth/login' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{'"
echo "    \"email\": \"$USER_EMAIL\","
echo "    \"password\": \"$USER_PASSWORD\""
echo "  }'"
echo ""
echo "Save the 'access_token' value for use in other requests."
echo ""

echo "5. Get Current User Info"
echo "curl -X GET '$BASE_URL/auth/me' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
echo ""

# ============================================================================
# FILE MANAGEMENT ENDPOINTS
# ============================================================================

echo ""
echo "📁 FILE MANAGEMENT ENDPOINTS"
echo "============================="
echo ""

echo "6. Upload File"
echo "curl -X POST '$BASE_URL/files/upload' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \\"
echo "  -F 'file=@/path/to/document.pdf'"
echo ""
echo "Supported formats: PDF, TXT, DOCX"
echo "Max file size: 50MB"
echo ""

echo "7. List User Files"
echo "curl -X GET '$BASE_URL/files/list?skip=0&limit=100' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
echo ""

echo "8. Sync Embeddings for a File"
echo "curl -X POST '$BASE_URL/files/sync-embeddings/FILE_ID' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
echo ""
echo "This generates embeddings and stores them in Qdrant"
echo ""

echo "9. Delete File"
echo "curl -X DELETE '$BASE_URL/files/FILE_ID' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
echo ""

# ============================================================================
# CHAT ENDPOINTS (RAG + Agent)
# ============================================================================

echo ""
echo "💬 CHAT ENDPOINTS (RAG + AGENT)"
echo "================================="
echo ""

echo "10. Ask a Question (RAG + ReAct Agent)"
echo "curl -X POST '$BASE_URL/chat/ask' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{'"
echo "    \"question\": \"What is the main topic of the document?\""
echo "  }'"
echo ""

echo "11. Get Chat History"
echo "curl -X GET '$BASE_URL/chat/history?skip=0&limit=50' \\"
echo "  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
echo ""

# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

echo ""
echo "📋 COMPLETE WORKFLOW EXAMPLE"
echo "============================="
echo ""

echo "Step 1: Register"
echo "  curl -X POST '$BASE_URL/auth/register' -H 'Content-Type: application/json' -d '{\"email\":\"user@test.com\",\"password\":\"pass123\"}'"
echo ""

echo "Step 2: Login (Copy access_token)"
echo "  curl -X POST '$BASE_URL/auth/login' -H 'Content-Type: application/json' -d '{\"email\":\"user@test.com\",\"password\":\"pass123\"}'"
echo ""

echo "Step 3: Create test file"
echo "  echo 'Test content about AI and RAG' > test.txt"
echo ""

echo "Step 4: Upload file (Copy file_id)"
echo "  curl -X POST '$BASE_URL/files/upload' -H 'Authorization: Bearer ACCESS_TOKEN' -F 'file=@test.txt'"
echo ""

echo "Step 5: Sync embeddings"
echo "  curl -X POST '$BASE_URL/files/sync-embeddings/FILE_ID' -H 'Authorization: Bearer ACCESS_TOKEN'"
echo ""

echo "Step 6: Ask a question"
echo "  curl -X POST '$BASE_URL/chat/ask' -H 'Authorization: Bearer ACCESS_TOKEN' -H 'Content-Type: application/json' -d '{\"question\":\"What is the content about?\"}'"
echo ""

# ============================================================================
# USEFUL CURL OPTIONS
# ============================================================================

echo ""
echo "🔧 USEFUL CURL OPTIONS"
echo "======================"
echo ""

echo "Pretty-print JSON response:"
echo "  curl ... | python3 -m json.tool"
echo "  OR"
echo "  curl ... | jq ."
echo ""

echo "Save to file:"
echo "  curl ... -o response.json"
echo ""

echo "Show response headers:"
echo "  curl -i ..."
echo ""

echo "Show request and response:"
echo "  curl -v ..."
echo ""

echo "Set timeout:"
echo "  curl --max-time 30 ..."
echo ""

echo "Custom headers:"
echo "  curl -H 'X-Custom-Header: value' ..."
echo ""

# ============================================================================
# IMPORTANT NOTES
# ============================================================================

echo ""
echo "📌 IMPORTANT NOTES"
echo "=================="
echo ""

echo "1. Always include Authorization header with Bearer token"
echo "2. Replace YOUR_ACCESS_TOKEN with actual token from login response"
echo "3. Replace FILE_ID with actual file ID from upload response"
echo "4. Use proper quotes for JSON data"
echo "5. File paths in -F option can be relative or absolute"
echo ""

# ============================================================================
# ERROR HANDLING
# ============================================================================

echo ""
echo "⚠️  ERROR HANDLING"
echo "=================="
echo ""

echo "400 Bad Request"
echo "  - Invalid request format or missing required fields"
echo ""

echo "401 Unauthorized"
echo "  - Missing or invalid access token"
echo "  - Token has expired"
echo ""

echo "403 Forbidden"
echo "  - User trying to access another user's resources"
echo ""

echo "404 Not Found"
echo "  - Resource doesn't exist"
echo ""

echo "413 Request Entity Too Large"
echo "  - Uploaded file exceeds size limit"
echo ""

echo "422 Unprocessable Entity"
echo "  - Validation error (e.g., invalid email format)"
echo ""

echo "500 Internal Server Error"
echo "  - Server-side error"
echo ""

# ============================================================================
# POSTMAN COLLECTION
# ============================================================================

echo ""
echo "📧 POSTMAN COLLECTION"
echo "====================="
echo ""

echo "Import file: Chat-RAG-AI-Agent.postman_collection.json"
echo "Then set environment variables:"
echo "  - base_url = http://localhost:8000"
echo "  - access_token = (from login response)"
echo "  - file_id = (from upload response)"
echo ""

# ============================================================================
# PYTHON EXAMPLE
# ============================================================================

echo ""
echo "🐍 PYTHON CLIENT EXAMPLE"
echo "========================"
echo ""

echo "See example_usage.py for complete Python implementation"
echo "Usage: python example_usage.py"
echo ""

# ============================================================================
# DEBUGGING
# ============================================================================

echo ""
echo "🐛 DEBUGGING TIPS"
echo "=================="
echo ""

echo "Check API is running:"
echo "  curl http://localhost:8000/health/"
echo ""

echo "Check all services:"
echo "  curl http://localhost:8000/health/detailed"
echo ""

echo "View Docker logs:"
echo "  docker-compose logs -f api"
echo ""

echo "Access Swagger UI:"
echo "  http://localhost:8000/docs"
echo ""

echo "Access ReDoc:"
echo "  http://localhost:8000/redoc"
echo ""

# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

echo ""
echo "⚡ PERFORMANCE TESTING"
echo "======================"
echo ""

echo "Run 100 requests:"
echo "  for i in {1..100}; do curl http://localhost:8000/health/; done"
echo ""

echo "Measure response time:"
echo "  time curl -X POST http://localhost:8000/chat/ask -H 'Authorization: Bearer TOKEN' -H 'Content-Type: application/json' -d '{\"question\":\"test\"}'"
echo ""

echo "Load testing with Apache Bench:"
echo "  ab -n 1000 -c 10 http://localhost:8000/health/"
echo ""

# ============================================================================
# ADVANCED SCENARIOS
# ============================================================================

echo ""
echo "🚀 ADVANCED SCENARIOS"
echo "====================="
echo ""

echo "Upload multiple files in loop:"
echo "  for file in *.pdf; do curl -X POST http://localhost:8000/files/upload -H 'Authorization: Bearer TOKEN' -F \"file=@\$file\"; done"
echo ""

echo "Sync all files automatically:"
echo "  # First get all file IDs, then loop and sync each"
echo ""

echo "Ask multiple questions and collect answers:"
echo "  # Use for creating test datasets"
echo ""

echo "Export chat history:"
echo "  curl http://localhost:8000/chat/history -H 'Authorization: Bearer TOKEN' | jq . > chat_history.json"
echo ""

echo ""
echo "✅ Ready to test! Copy any command and run it in terminal."
echo ""
