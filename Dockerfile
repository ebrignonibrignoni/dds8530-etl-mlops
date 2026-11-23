FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY etl/ etl/
COPY api/ api/
COPY models/ models/
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api.app:app"]
