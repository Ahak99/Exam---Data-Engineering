# Use the official Python image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y python3-dev && \
    apt-get install -y libblas-dev liblapack-dev && \
    apt-get install -y libatlas-base-dev gfortran && \
    apt-get clean

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set the default command to run your application
CMD python main.py
