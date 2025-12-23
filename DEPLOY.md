# 🚀 Deployment Guide

This guide covers deploying PaperFeeder for daily automated execution using GitHub Actions, or running it on your own server.

---

## 📋 Table of Contents

- [Deployment Options](#deployment-options)
- [Option 1: GitHub Actions (Recommended)](#option-1-github-actions-recommended)
- [Option 2: Cron Job (Server)](#option-2-cron-job-server)
- [Option 3: Docker](#option-3-docker)
- [Configuration Best Practices](#configuration-best-practices)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Deployment Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **GitHub Actions** | Free, automated, no server needed | 6-hour runtime limit | Most users |
| **Cron Job** | Full control, no limits | Requires server | Power users |
| **Docker** | Portable, reproducible | Requires container knowledge | Teams |

---

## Option 1: GitHub Actions (Recommended)

### Step 1: Fork the Repository

1. Go to https://github.com/gaoxin492/PaperFeeder
2. Click "Fork" in the top-right corner

### Step 2: Set Up Secrets

Go to your forked repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add the following secrets:

```
LLM_API_KEY              # Your main LLM API key (required)
LLM_BASE_URL             # e.g., https://api.openai.com/v1
LLM_MODEL                # e.g., gpt-4o-mini

LLM_FILTER_API_KEY       # Filter LLM key (optional, can reuse LLM_API_KEY)
LLM_FILTER_BASE_URL      # e.g., https://api.openai.com/v1
LLM_FILTER_MODEL         # e.g., gpt-4o-mini

TAVILY_API_KEY           # Tavily search API (optional but recommended)

RESEND_API_KEY           # Email delivery (optional)
EMAIL_FROM               # e.g., papers@resend.dev
EMAIL_TO                 # Your email address
```

### Step 3: Enable GitHub Actions

1. Go to **Actions** tab in your repo
2. Click "I understand my workflows, go ahead and enable them"

### Step 4: Configure Schedule

Edit `.github/workflows/daily-paper.yml`:

```yaml
name: Daily Paper Digest

on:
  schedule:
    # Run at 8:00 AM UTC (adjust to your timezone)
    - cron: '0 8 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  fetch-and-summarize:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run PaperFeeder
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
          LLM_MODEL: ${{ secrets.LLM_MODEL }}
          LLM_FILTER_API_KEY: ${{ secrets.LLM_FILTER_API_KEY }}
          LLM_FILTER_BASE_URL: ${{ secrets.LLM_FILTER_BASE_URL }}
          LLM_FILTER_MODEL: ${{ secrets.LLM_FILTER_MODEL }}
          TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
          EMAIL_TO: ${{ secrets.EMAIL_TO }}
        run: |
          python main.py
      
      - name: Upload report (if email fails)
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: paper-digest
          path: report_preview.html
```

### Step 5: Test

Click **Actions** → **Daily Paper Digest** → **Run workflow** to test manually.

### Schedule Options

```yaml
# Daily at 8 AM UTC
- cron: '0 8 * * *'

# Weekdays only at 9 AM UTC
- cron: '0 9 * * 1-5'

# Twice daily (8 AM and 8 PM UTC)
- cron: '0 8,20 * * *'

# Weekly on Monday at 9 AM UTC
- cron: '0 9 * * 1'
```

Convert your timezone: https://crontab.guru/

---

## Option 2: Cron Job (Server)

### Step 1: Clone Repository

```bash
ssh your-server
cd /opt
git clone https://github.com/gaoxin492/PaperFeeder.git
cd PaperFeeder
```

### Step 2: Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
# LLM Settings
LLM_API_KEY=sk-xxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

LLM_FILTER_API_KEY=sk-xxxxx
LLM_FILTER_BASE_URL=https://api.openai.com/v1
LLM_FILTER_MODEL=gpt-4o-mini

# Research
TAVILY_API_KEY=tvly-xxxxx

# Email
RESEND_API_KEY=re-xxxxx
EMAIL_FROM=papers@resend.dev
EMAIL_TO=your@email.com
EOF

chmod 600 .env
```

### Step 4: Create Run Script

```bash
cat > run_paperfeeder.sh << 'EOF'
#!/bin/bash
cd /opt/PaperFeeder
source venv/bin/activate
source .env
python main.py >> logs/paperfeeder.log 2>&1
EOF

chmod +x run_paperfeeder.sh
mkdir -p logs
```

### Step 5: Set Up Cron

```bash
crontab -e
```

Add:

```cron
# Daily at 8 AM
0 8 * * * /opt/PaperFeeder/run_paperfeeder.sh

# Or with full path
0 8 * * * cd /opt/PaperFeeder && /opt/PaperFeeder/venv/bin/python main.py >> logs/paperfeeder.log 2>&1
```

### Step 6: Test

```bash
./run_paperfeeder.sh
tail -f logs/paperfeeder.log
```

### Log Rotation (Optional)

```bash
cat > /etc/logrotate.d/paperfeeder << 'EOF'
/opt/PaperFeeder/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

---

## Option 3: Docker

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run
CMD ["python", "main.py"]
```

### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  paperfeeder:
    build: .
    env_file: .env
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./manual_papers.json:/app/manual_papers.json
      - ./logs:/app/logs
    restart: unless-stopped
```

### Step 3: Build and Run

```bash
# Build
docker-compose build

# Test run
docker-compose run --rm paperfeeder python main.py --dry-run

# Run in background
docker-compose up -d
```

### Step 4: Schedule with Cron

```bash
# In host crontab
0 8 * * * cd /opt/PaperFeeder && docker-compose run --rm paperfeeder
```

Or use a scheduler container like [ofelia](https://github.com/mcuadros/ofelia).

---

## 📝 Configuration Best Practices

### 1. API Key Management

**DO:**
- Use environment variables or GitHub Secrets
- Use `.env` file locally (add to `.gitignore`)
- Rotate keys periodically

**DON'T:**
- Commit API keys to Git
- Share keys in public channels
- Use same key for dev and prod

### 2. Cost Optimization

```bash
# Use cheap model for filtering (called 2x per paper)
LLM_FILTER_MODEL=gpt-4o-mini

# Use better model for summarization (called once)
LLM_MODEL=gpt-4o

# Limit papers to save costs
# In config.yaml:
max_papers: 3  # Only summarize top 3
```

### 3. Rate Limiting

```yaml
# In config.yaml
llm_filter_threshold: 10  # Only use LLM filter if >10 papers
```

### 4. Monitoring

```bash
# Add to run script
if [ $? -ne 0 ]; then
  echo "PaperFeeder failed" | mail -s "Alert" admin@example.com
fi

# Or use external monitoring
curl -m 10 --retry 3 https://hc-ping.com/your-uuid
```

---

## 🔧 Troubleshooting

### GitHub Actions Fails

**Symptom**: Workflow shows red X

**Check**:
1. Go to Actions tab → Click failed workflow
2. Check logs for error messages
3. Common issues:
   - Missing secrets
   - Invalid API keys
   - Timeout (increase `timeout-minutes`)

**Fix**:
```bash
# Test locally first
python main.py --dry-run

# Check secrets are set correctly
# Settings → Secrets → Check all required secrets exist
```

### Email Not Sending

**Symptom**: Script succeeds but no email

**Check**:
```bash
# Test email configuration
python -c "
from emailer import ResendEmailer
import asyncio
import os

async def test():
    emailer = ResendEmailer(
        api_key=os.getenv('RESEND_API_KEY'),
        from_email=os.getenv('EMAIL_FROM', 'papers@resend.dev')
    )
    result = await emailer.send(
        to=os.getenv('EMAIL_TO'),
        subject='Test Email',
        html_content='<h1>Test</h1>'
    )
    print('Success!' if result else 'Failed')

asyncio.run(test())
"
```

**Fix**:
- Verify `RESEND_API_KEY` is correct
- Use `papers@resend.dev` as `EMAIL_FROM`
- Check spam folder
- Or use `--dry-run` to save to file

### Tavily API 401 Error

**Symptom**: `Unauthorized: missing or invalid API key`

**Fix**:
```bash
# Option 1: Get Tavily API key from https://tavily.com
export TAVILY_API_KEY=tvly-xxxxx

# Option 2: Run without Tavily (uses mock data)
unset TAVILY_API_KEY
python main.py --dry-run
```

### Out of Memory

**Symptom**: Process killed or hangs

**Fix**:
```yaml
# In config.yaml, reduce batch sizes
llm_filter_threshold: 10  # Process fewer papers
max_papers: 3  # Summarize fewer papers

# Limit PDF pages
pdf_max_pages: 10  # Only first 10 pages
```

### arXiv Timeout

**Symptom**: `Timeout on attempt 1/3`

**Fix**:
```bash
# arXiv API can be slow (10-60s), be patient
# Or reduce categories in config.yaml:
arxiv_categories:
  - cs.LG  # Just one category for faster queries
```

---

## 🔒 Security Checklist

- [ ] All API keys stored as secrets (not in code)
- [ ] `.env` file in `.gitignore`
- [ ] GitHub repo is private (if using GitHub Actions)
- [ ] Email credentials not exposed in logs
- [ ] Regular key rotation schedule
- [ ] Monitoring enabled for failures

---

## 📊 Performance Tips

### Speed Up Execution

1. **Reduce arXiv categories**:
   ```yaml
   arxiv_categories:
     - cs.LG  # Just one instead of 5
   ```

2. **Disable PDF processing**:
   ```yaml
   extract_fulltext: false
   ```

3. **Use local LLM**:
   ```bash
   # Install Ollama
   export LLM_BASE_URL=http://localhost:11434/v1
   export LLM_MODEL=llama3
   ```

### Reduce Costs

1. **Use cheaper models**:
   ```bash
   LLM_FILTER_MODEL=gpt-4o-mini  # $0.15/1M tokens
   LLM_MODEL=gpt-4o-mini          # Same model for both
   ```

2. **Limit papers**:
   ```yaml
   max_papers: 3  # Only top 3
   ```

3. **Disable Tavily**:
   ```bash
   # Free tier: 1000 searches/month
   # If you exceed, unset TAVILY_API_KEY
   ```

---

## 🆘 Support

- **GitHub Issues**: https://github.com/gaoxin492/PaperFeeder/issues
- **Discussions**: https://github.com/gaoxin492/PaperFeeder/discussions
- **Email**: Check repo for contact info

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cron Expression Generator](https://crontab.guru/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Resend Documentation](https://resend.com/docs)
- [Tavily Documentation](https://docs.tavily.com/)

---

**🎉 Happy Deploying!**

Once deployed, you'll receive daily paper digests automatically. Adjust `config.yaml` to tune filtering and research interests to your liking.