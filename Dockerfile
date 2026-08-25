FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY backend/ backend/
COPY static/ static/

EXPOSE 8765

# Learner data lives in the browser; the container only generates content
# and caches audio under /app/runtime (mount a volume to persist the cache).
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8765"]
