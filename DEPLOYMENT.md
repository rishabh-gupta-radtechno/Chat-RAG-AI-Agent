"""
Deployment guide for Chat RAG AI Agent
"""

# Deployment Guides

## 🚀 Local Development

See [QUICKSTART.md](QUICKSTART.md) for quick setup.

## 🐳 Docker Deployment (Recommended for Testing)

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum, 8GB recommended

### Steps

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd Chat-RAG-AI-Agent
   ```

2. **Configure environment**
   ```bash
   cp .env .env.prod
   # Edit .env.prod with your settings
   # - Set DEBUG=False
   # - Update SECRET_KEY
   # - Configure CORS origins
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Verify health**
   ```bash
   curl http://localhost:8000/health/detailed
   ```

## ☁️ AWS Deployment (ECS + RDS)

### Architecture
- **ECS Fargate**: FastAPI container
- **RDS PostgreSQL**: Managed database
- **Qdrant**: Self-managed or Qdrant Cloud
- **ALB**: Application Load Balancer
- **CloudWatch**: Logging and monitoring

### Steps

1. **Push Docker image to ECR**
   ```bash
   aws ecr create-repository --repository-name chat-rag-api
   docker build -t chat-rag-api .
   docker tag chat-rag-api:latest <aws_account>.dkr.ecr.<region>.amazonaws.com/chat-rag-api:latest
   docker push <aws_account>.dkr.ecr.<region>.amazonaws.com/chat-rag-api:latest
   ```

2. **Create RDS PostgreSQL instance**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier chat-rag-db \
     --engine postgres \
     --db-instance-class db.t3.micro \
     --allocated-storage 100 \
     --master-username postgres \
     --master-user-password <strong-password>
   ```

3. **Set up ECS**
   - Create ECS cluster
   - Create task definition (update image URI from ECR)
   - Create service (expose port 8000)
   - Attach ALB

4. **Configure environment variables**
   - Update `DATABASE_URL` to RDS endpoint
   - Update `QDRANT_URL` to Qdrant Cloud or self-managed instance
   - Update `OLLAMA_BASE_URL`
   - Set `SECRET_KEY`

5. **Update security groups**
   - ALB security group: Allow port 80/443
   - ECS security group: Allow port 8000 from ALB
   - RDS security group: Allow port 5432 from ECS
   - Qdrant security group: Allow port 6333 from ECS

## 🔗 AWS Lambda Deployment (Alternative)

For serverless deployment, consider:
- API Gateway → Lambda (with long timeout)
- RDS Proxy for database connection pooling
- Limitations: Cold starts, timeout constraints

## 🔄 Kubernetes Deployment

### Prerequisites
- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+

### Basic Setup

1. **Create secrets**
   ```bash
   kubectl create secret generic chat-rag-secrets \
     --from-literal=database-url=postgresql://... \
     --from-literal=secret-key=... \
     --from-literal=qdrant-url=...
   ```

2. **Create ConfigMap**
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: chat-rag-config
   data:
     OLLAMA_CHAT_MODEL: "qwen3:8b"
     OLLAMA_EMBEDDING_MODEL: "bge-m3"
     ENV: "production"
   ```

3. **Deploy application**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: chat-rag-api
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: chat-rag-api
     template:
       metadata:
         labels:
           app: chat-rag-api
       spec:
         containers:
         - name: api
           image: <ecr-uri>/chat-rag-api:latest
           ports:
           - containerPort: 8000
           env:
           - name: DATABASE_URL
             valueFrom:
               secretKeyRef:
                 name: chat-rag-secrets
                 key: database-url
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
   ```

4. **Create service**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: chat-rag-api
   spec:
     type: LoadBalancer
     ports:
     - port: 80
       targetPort: 8000
     selector:
       app: chat-rag-api
   ```

## 🔒 Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Update `SECRET_KEY` to 32+ character random string
- [ ] Configure strong database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure CORS origins (not "*")
- [ ] Enable database backups
- [ ] Set up monitoring and alerting
- [ ] Configure rate limiting
- [ ] Set up CI/CD pipeline
- [ ] Implement request validation
- [ ] Add API authentication (API key, OAuth2)
- [ ] Enable logging to external service (CloudWatch, DataDog, etc.)
- [ ] Set up health checks and auto-recovery
- [ ] Configure secrets management
- [ ] Set up database connection pooling
- [ ] Test disaster recovery

## 📊 Monitoring Setup

### CloudWatch (AWS)
```bash
# View logs
aws logs tail /ecs/chat-rag-api --follow

# Set up alarms
aws cloudwatch put-metric-alarm \
  --alarm-name ChatRAGErrorRate \
  --metric-name Errors \
  --namespace AWS/ECS
```

### Sentry (Error Tracking)
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="<sentry-dsn>",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)
```

### DataDog
```python
from datadog import initialize, api
from datadog.api import Monitor

options = {'api_key': '<api-key>', 'app_key': '<app-key>'}
initialize(**options)
```

## 🔄 CI/CD Pipeline Example (GitHub Actions)

```yaml
name: Deploy Chat RAG API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest --cov=app

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and push Docker image
        run: |
          docker build -t chat-rag-api:${{ github.sha }} .
          docker push <ecr-uri>/chat-rag-api:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster chat-rag-cluster \
            --service chat-rag-api \
            --force-new-deployment
```

## 🔐 SSL/TLS Setup

### Using Let's Encrypt with Nginx
```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Scaling Considerations

- **Horizontal Scaling**: Add more API instances behind load balancer
- **Database Scaling**: Use read replicas for read operations
- **Caching**: Implement Redis caching for frequent queries
- **Vector DB Scaling**: Use Qdrant clustering
- **Content Delivery**: Use CDN for static files

## 🆘 Troubleshooting Deployment

### 404 on health check
- Verify service is running: `docker-compose ps`
- Check logs: `docker-compose logs api`
- Verify port mapping: `docker ps`

### Database connection errors
- Check RDS security group rules
- Verify database credentials
- Test connection: `psql -h <rds-endpoint> -U postgres`

### Ollama inference failing
- Check Ollama service status
- Verify model is loaded: `curl http://ollama:11434/api/tags`
- Check available resources

### Qdrant collection errors
- Verify Qdrant is running and accessible
- Check Qdrant web UI: http://qdrant:6333/dashboard
- Clear data if needed: `docker-compose down -v`

---

For more details, see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
