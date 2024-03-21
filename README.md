# AvesEcho v1


##  Introduction

This repo contains AvesEcho v1 model and scripts for processing a batch of audio data. AvesEcho v1 is a multi-label bird species recognition model, developed for Europe. 
We will keep this repository up-to-date with new models and improved interfaces to enable people to run the analysis.



## Building the model (using docker)

Install Docker for Ubuntu:

```
sudo apt update
sudo apt install docker.io
```

Once you have Docker installed on your system, you can go ahead and clone the AvesEcho repo. After cloning, move into the repository directory and build the Docker image using:
```
sudo docker build -t avesechov1 .
```
## Running the AvesEcho Service with Docker Compose

AvesEcho is containerized for easy deployment and execution. To run the service, you can use the provided `docker-compose.yml` file, which defines the necessary settings and parameters for the Docker container.

### Docker Compose Configuration

Below are key configurations in the `docker-compose.yml` file:

- `image: avesechov1`: This specifies the name of the Docker image to use for the container.
- `command`: This defines the arguments that accompany the command that will run when the container starts up (defined in `start.sh`):
  - `--mconf`: Sets the confidence threshold for the model. If set to `None` (default), pre-computed species-wise thresholds are used. For no thresholds set to `0`.
  - `--add_csv`: Enables output in CSV format in addition to the JSON output.
  - `--add_filtering`: Adds location based filtering emploting the `flist`.
  - `--flist`: a species list needs to be supplied if adding the `--add_filtering` flag. 
  - `--i`: Sets the input directory for audio files within the container.
  - `--lat`: Latitude for geographic filtering. Ignores the `flist`.
  - `--lon`: Longitude for geographic filtering. Ignores the `flist`.
  - `--maxpool`: Use model for generating temporally-summarised output.
  - `--model_name`: Name of the model to use (fc - CNN/ passt - Transformer-based; by default the classifier is set to `fc`).
- `deploy.resources.reservations.devices`: Reserves GPU resources for the container, allowing for GPU acceleration in processing.
- `volumes`: Mounts the local directory to the container for persistent storage and access to data:
  - `/home/burooj/models/avesecho-v1/outputs:/app/outputs`: Mounts the host outputs directory to the container's working directory. Make sure to point it to your own host outputs directory. Note that, this can be a directory located outside the model directory on your host machine.  
- `shm_size: 4g`: Allocates 4GB of shared memory to the container, which is necessary for memory-intensive operations.

### Running the Container

To start the service with Docker Compose, run the following command in the terminal from the same directory as your `docker-compose.yml`:

```
sudo docker compose up

```

On successful execution the results will be stored in `/outputs`. 

To run the service without Docker Compose, you may run the following command (for cpu only support):

```
sudo docker run -v .:/app avesechov1 --i audio/
```

And, the following command for GPU availability:

```
sudo docker run --gpus all -v .:/app avesechov1 --i audio/
```
To open your container in an interactive mode with a bash shell, use the following command:

```
sudo docker run --gpus all -it --entrypoint /bin/bash avesechov1
```


