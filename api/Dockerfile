# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /api
WORKDIR /api

# Copy the requirements file into the container
COPY api/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install pb-buddy custom code
COPY pb_buddy pb_buddy
COPY poetry.lock pyproject.toml pb-buddy
# RUN pip install -e pb-buddy/

# Copy the application code into the container
COPY api/.env api/main.py api/model.py api/predictor.py .

# Copy model folder into container
COPY api/assets assets

# Set the environment variable to use the .env file
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

# Expose the port that the application will run on
EXPOSE 80

# Run the application with uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "80", "main:app"]
