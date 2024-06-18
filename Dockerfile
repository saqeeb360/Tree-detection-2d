# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /myapp

# Copy the current directory contents into the container at /myapp
# COPY . .
COPY requirements.txt ./
COPY *.py ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data
RUN mkdir -p /myapp/raw_data
RUN mkdir -p /myapp/train_data
RUN mkdir -p /myapp/test_data
RUN mkdir -p /myapp/test_results
RUN mkdir -p /myapp/model_logs

# Set environment variables for NVIDIA GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Define the default command to run when starting the container
CMD ["python3", "preprocess.py"]

# Build the docker file
# docker build -t treedetection:1 .

# preprocess
# docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 preprocess.py

# train
# docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 train.py --epoch 100

# inference
# docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 inference.py