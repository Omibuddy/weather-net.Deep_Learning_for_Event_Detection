# Base image with Python and GPU support
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Set working directory inside container
WORKDIR /app

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Copy files into the container
COPY . .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Set the default command to run the training pipeline
CMD ["python3", "heatwave_training_pipeline.py"]
