FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements-dockers.txt ./

RUN pip install -r requirements-dockers.txt

COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]