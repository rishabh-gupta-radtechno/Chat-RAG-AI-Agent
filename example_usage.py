"""
Example usage of the Chat RAG AI Agent API
"""

import asyncio
import httpx
import json
from typing import Optional

BASE_URL = "http://localhost:8000"


class ChatRAGClient:
    """Client for Chat RAG AI Agent API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=30.0)

    async def register(self, email: str, password: str) -> dict:
        """Register a new user."""
        response = await self.client.post(
            f"{self.base_url}/auth/register",
            json={"email": email, "password": password},
        )
        response.raise_for_status()
        return response.json()

    async def login(self, email: str, password: str) -> dict:
        """Login and store access token."""
        response = await self.client.post(
            f"{self.base_url}/auth/login",
            json={"email": email, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        self.access_token = data["access_token"]
        return data

    async def upload_file(self, filepath: str) -> dict:
        """Upload a file."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        with open(filepath, "rb") as f:
            files = {"file": f}
            response = await self.client.post(
                f"{self.base_url}/files/upload",
                files=files,
                headers=headers,
            )
        response.raise_for_status()
        return response.json()

    async def list_files(self) -> list:
        """List user files."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = await self.client.get(
            f"{self.base_url}/files/list",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def sync_embeddings(self, file_id: str) -> dict:
        """Sync embeddings for a file."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = await self.client.post(
            f"{self.base_url}/files/sync-embeddings/{file_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def ask_question(self, question: str) -> dict:
        """Ask a question with RAG."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = await self.client.post(
            f"{self.base_url}/chat/ask",
            json={"question": question},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_chat_history(self, skip: int = 0, limit: int = 50) -> list:
        """Get chat history."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = await self.client.get(
            f"{self.base_url}/chat/history",
            params={"skip": skip, "limit": limit},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


async def main():
    """Example usage."""
    client = ChatRAGClient()

    try:
        # 1. Register user
        print("1. Registering user...")
        user = await client.register("test@example.com", "testpassword123")
        print(f"✅ Registered: {user['email']}")

        # 2. Login
        print("\n2. Logging in...")
        tokens = await client.login("test@example.com", "testpassword123")
        print(f"✅ Login successful, token: {tokens['access_token'][:20]}...")

        # 3. List files (empty)
        print("\n3. Listing files (should be empty)...")
        files = await client.list_files()
        print(f"✅ Files: {len(files)} files found")

        # 4. Upload a file
        print("\n4. Uploading file...")
        # Create a test file first
        with open("test_document.txt", "w") as f:
            f.write("""
            This is a test document for the Chat RAG AI Agent.
            
            Key Information:
            - The system uses RAG (Retrieval-Augmented Generation)
            - It processes PDF, TXT, and DOCX files
            - Questions are answered using relevant document chunks
            - Responses include source references
            
            The AI Agent follows the ReAct pattern:
            1. Reason: Analyze the question and context
            2. Act: Retrieve relevant information
            3. Respond: Generate answer with citations
            """)

        upload_result = await client.upload_file("test_document.txt")
        file_id = upload_result["id"]
        print(f"✅ Uploaded: {upload_result['filename']} (ID: {file_id})")

        # 5. Sync embeddings
        print("\n5. Syncing embeddings...")
        sync_result = await client.sync_embeddings(file_id)
        print(f"✅ Synced: {sync_result['chunks_created']} chunks created")

        # 6. Ask a question
        print("\n6. Asking a question...")
        question = "What is the ReAct pattern?"
        answer = await client.ask_question(question)
        print(f"Question: {question}")
        print(f"Answer: {answer['answer']}")
        print(f"Sources: {len(answer['sources'])} sources found")
        for source in answer["sources"]:
            print(f"  - {source['filename']} (relevance: {source['relevance_score']:.2f})")

        # 7. Get chat history
        print("\n7. Getting chat history...")
        history = await client.get_chat_history()
        print(f"✅ Chat history: {len(history)} conversations")

        # Clean up
        import os
        os.remove("test_document.txt")

    finally:
        await client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
