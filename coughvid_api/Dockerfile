FROM python:3.8.6-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY models_coughvid_model.h5 /models_coughvid_model.h5

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.coughvid_api:app --host 0.0.0.0 --port $PORT
