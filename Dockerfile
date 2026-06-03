FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Tor if you want live fetching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory (if using persistent disk, adjust)
RUN mkdir -p /data

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
