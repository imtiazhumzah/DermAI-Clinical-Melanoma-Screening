# 1. Use a lean Python base image
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install necessary system dependencies for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
# Note: Using the --no-cache-dir flag keeps the image size small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project files into the container
# This includes app.py, model_utils.py, and your .pth weights
COPY . .

# 6. Set up a non-root user for Hugging Face security
# Hugging Face Spaces require the container to run as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 7. Expose the port Gradio uses
EXPOSE 7860

# 8. Define the command to run your app
# We use app.py as the entry point
CMD ["python", "app.py"]