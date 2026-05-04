#!/bin/bash

# Chat RAG AI Agent - API Testing Script

set -e

BASE_URL="${1:-http://localhost:8000}"
USER_EMAIL="testuser@example.com"
USER_PASSWORD="testpassword123"

echo "🧪 Chat RAG AI Agent - API Testing"
echo "=================================="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo "1️⃣  Testing Health Check..."
curl -s "$BASE_URL/health/" | python3 -m json.tool
echo ""

# Test 2: Register User
echo "2️⃣  Registering User..."
REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$USER_EMAIL\",
    \"password\": \"$USER_PASSWORD\"
  }")

echo "$REGISTER_RESPONSE" | python3 -m json.tool
USER_ID=$(echo "$REGISTER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo ""

# Test 3: Login User
echo "3️⃣  Logging in User..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$USER_EMAIL\",
    \"password\": \"$USER_PASSWORD\"
  }")

echo "$LOGIN_RESPONSE" | python3 -m json.tool
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
echo ""

# Test 4: Get Current User
echo "4️⃣  Getting Current User Info..."
curl -s -X GET "$BASE_URL/auth/me" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -m json.tool
echo ""

# Test 5: List Files (should be empty)
echo "5️⃣  Listing User Files (should be empty)..."
curl -s -X GET "$BASE_URL/files/list" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -m json.tool
echo ""

# Test 6: Detailed Health Check
echo "6️⃣  Detailed Health Check..."
curl -s "$BASE_URL/health/detailed" | python3 -m json.tool
echo ""

echo "✅ All tests completed!"
echo ""
echo "💡 To test file upload and RAG:"
echo "1. Create a test file: echo 'Test content' > test.txt"
echo "2. Upload: curl -X POST '$BASE_URL/files/upload' -H 'Authorization: Bearer $ACCESS_TOKEN' -F 'file=@test.txt'"
echo "3. Sync embeddings: curl -X POST '$BASE_URL/files/sync-embeddings/{file_id}' -H 'Authorization: Bearer $ACCESS_TOKEN'"
echo "4. Ask question: curl -X POST '$BASE_URL/chat/ask' -H 'Authorization: Bearer $ACCESS_TOKEN' -H 'Content-Type: application/json' -d '{\"question\": \"What is in the file?\"}'"
