# Building and Running the Docker Container

This README provides instructions for building and running the Docker container for the FastAPI application in the `impute-code` directory.

## Prerequisites

- **Project Files**: Ensure the following files are in the `impute-code` directory:
  - `Dockerfile`
  - `main.py`



## Building the Docker Image

1. Open a terminal and navigate to the project directory:
   ```bash
   cd impute-code
   ```
2. Build the Docker image:
   ```bash
   docker build -t impute-code .
   ```
3. Verify the image:
   ```bash
   docker images
   ```

## Running the Docker Container

1. Start the container:
   ```bash
   docker run -d -p 8000:8000 --name impute-container impute-code
   ```
2. Verify the container is running:
   ```bash
   docker ps
   ```
3. Check logs:
   ```bash
   docker logs impute-container
   ```

## Stopping and Cleaning Up

- Stop the container:
  ```bash
  docker stop impute-container
  ```
- Remove the container:
  ```bash
  docker rm impute-container
  ```
- Remove the image (optional):
  ```bash
  docker rmi impute-code
  ```