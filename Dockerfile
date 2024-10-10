# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory inside the container to /code
WORKDIR /code

# Copy the requirements file from the local machine to the container
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies listed in the requirements file
# The --no-cache-dir option reduces the size of the final image by not caching the installation
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the app directory from the local machine to the container's /code/app directory
COPY ./app /code/app

# Expose port 8080 so the container can listen on it
EXPOSE 8080

# Command to run the application using uvicorn when the container starts
CMD ["uvicorn", "app.server.py:app", "--host", "0.0.0.0", "--port", "8080"]

## Stage 1: Build Stage
#FROM python:3.11 AS builder
#
## Copy the requirements file to install dependencies
#COPY ./requirements.txt .
#
## Install the Python dependencies listed in the requirements file
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy the app directory to the build context
#COPY ./app ./app
#
## Copy the .joblib file to the app directory
#COPY ./app/disease_classifier.joblib ./app/disease_classifier.joblib
#
## Stage 2: Final Stage
#FROM python:3.11-slim
#
## Copy only the necessary files from the build stage
#COPY --from=builder ./app ./app
#COPY --from=builder ./requirements.txt ./requirements.txt
#
## Expose port 8080 so the container can listen on it
#EXPOSE 8080
#
## Command to run the application using uvicorn when the container starts
#CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]

## Stage 1: Build Stage
#FROM python:3.11 AS builder
#
## Copy the requirements file to install dependencies
#COPY ./requirements.txt .
#
## Install the Python dependencies listed in the requirements file
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy the app directory to the build context
#COPY ./app ./app
#
## Stage 2: Final Stage
#FROM python:3.11-slim
#
## Copy only the necessary files from the build stage
#COPY --from=builder ./app ./app
#COPY --from=builder ./requirements.txt ./requirements.txt
#
## Expose port 8080 so the container can listen on it
#EXPOSE 8080
#
## Command to run the application using python -m uvicorn
#CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]



# Stage 1: Build Stage
#FROM python:3.11 AS builder
#
## Copy the requirements file to install dependencies
#COPY ./requirements.txt .
#
## Check the contents of requirements.txt
#RUN cat requirements.txt
#
## Install the Python dependencies listed in the requirements file
#RUN pip install --no-cache-dir -r requirements.txt
#
## Verify that uvicorn is installed
#RUN pip show uvicorn
#
## Copy the app directory to the build context
#COPY ./app ./app
#
## Stage 2: Final Stage
#FROM python:3.11-slim
#
## Copy only the necessary files from the build stage
#COPY --from=builder ./app ./app
#
## Install uvicorn if not already present
#RUN pip install --no-cache-dir uvicorn
#
## Expose port 8080 so the container can listen on it
#EXPOSE 8080
#
## Command to run the application using python -m uvicorn
#CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
#
## Stage 1: Build Stage
#FROM python:3.11 AS builder
#
## Copy the requirements file to install dependencies
#COPY ./requirements.txt .
#
## Check the contents of requirements.txt
#RUN cat requirements.txt
#
## Install the Python dependencies listed in the requirements file
#RUN pip install --no-cache-dir -r requirements.txt
#
## Verify that uvicorn is installed
#RUN pip show uvicorn
#
## Copy the app directory to the build context
#COPY ./app ./app
#
## Stage 2: Final Stage
#FROM python:3.11-slim
#
## Copy only the necessary files from the build stage
#COPY --from=builder ./app ./app
#COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
#
## Install only the required dependencies from requirements.txt
#COPY --from=builder /requirements.txt /requirements.txt
#RUN pip install --no-cache-dir -r /requirements.txt
#
## Expose port 8080 so the container can listen on it
#EXPOSE 8080
#
## Command to run the application using python -m uvicorn
#CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]



# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server (replace main:app with your FastAPI app)
CMD ["uvicorn", "server.py:app", "--host", "0.0.0.0", "--port", "8080"]
