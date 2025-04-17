# Use an official lightweight Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port
EXPOSE 8080

# Run the Flask app (update entry point)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.web_ui:app"]
