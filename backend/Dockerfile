# 1. Use a specific, slim version of Python to keep the image small
FROM python:3.11-slim

# 2. Set the working directory to a generic folder name like /app
WORKDIR /app

# 3. Copy just the requirements first (optimizes Docker build speed)
COPY requirements.txt ./

# 4. Install your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your local files (including your real app.py) into /app
COPY . .

# 6. Set the default fallback port
ENV PORT=5000

# 7. Start your Flask application
CMD ["python", "app.py"]
