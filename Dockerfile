# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi openenv-core

# Copy the rest of the code
COPY . .

# Expose the default port (HF Spaces/OpenEnv usually expect 7860)
EXPOSE 7860

# Start the uvicorn server serving server.py
# (Note: app.py is Gradios app, we will use server.app for the API endpoints)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
