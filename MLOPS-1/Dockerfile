# only use slim version to reduce image size

FROM python:3.14-slim

WORKDIR /app

 # . pastes the contents of the current directory into the container at /app
COPY requirements.txt .

# avoid caching of dependencies to ensure the latest versions are installed
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code into the container and save the model.pkl file in the same directory as main.py
COPY main.py .


EXPOSE 8000

# run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]