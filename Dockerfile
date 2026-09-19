FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., build-tools for scikit-learn/matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 8080 (Cloud Run requires listening on port 8080)
EXPOSE 8080

# Command to run Streamlit pointing to port 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]