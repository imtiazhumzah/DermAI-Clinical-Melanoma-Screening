# Use a lean Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
# We swapped libgl1-mesa-glx for libgl1 to fix the "Trixie" build error
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set up a non-root user for Hugging Face security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port Gradio uses
EXPOSE 7860

# Command to run your app
CMD ["python", "app.py"]