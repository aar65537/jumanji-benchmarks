# Base image
FROM docker.io/library/python:3.11-slim-bullseye

# Install python dependencies
RUN python -m pip install --no-cache-dir dash==2.10.2 pandas==2.0.2

# Set working directory
WORKDIR /home

# Copy files
COPY jumanji_benchmarks jumanji_benchmarks
COPY results results

# Set environment variables
ENV PYTHONPATH /home/jumanji_benchmarks:$PYTHONPATH

# Start dashboard
CMD python jumanji_benchmarks/dashboard.py