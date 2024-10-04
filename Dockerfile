FROM python:3.10-slim


# Set the working directory in the container
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip

# Install Python dependencies
RUN pip install --upgrade pip


# Install additional Python dependencies

RUN pip3 install torch 
RUN pip3 install boto3 pandas pyyaml langchain evaluate

# Copy the entrypoint script into the container.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy project files (including model operations and other necessary files)
COPY dfl_evaluation.py /app/dfl_evaluation.py



# Set the entry point for the container.
ENTRYPOINT ["/app/entrypoint.sh"]