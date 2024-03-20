# Start with a pytorch base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables to prevent interactive prompts
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and other necessary tools
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1


# Set the working directory
WORKDIR /app

# Copy the content of the local src directory to the working directory
COPY . /app

# Install other Python dependencies from requirements.txt
# Create the directory and download the passt file into it using curl
RUN pip3 install --no-cache-dir -r requirements.txt && \
    mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    curl -o /root/.cache/torch/hub/checkpoints/passt-s-kd-ap.486.pt -L https://github.com/kkoutini/PaSST/releases/download/v.0.0.9/passt-s-kd-ap.486.pt

# Set the ENTRYPOINT
ENTRYPOINT ["/app/start.sh"]
