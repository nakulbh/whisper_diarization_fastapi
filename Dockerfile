# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install git, ffmpeg, libsndfile1, and other necessary dependencies
RUN apt-get update && \
    apt-get install -y git ffmpeg libsndfile1 python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set environment variable to ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run FastAPI server
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "300", "--preload"]

