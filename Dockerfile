FROM python:3.11

WORKDIR /code

RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY ./pyproject.toml .
RUN pip install --no-cache-dir --upgrade .

COPY ./app /code/app
COPY ./data /code/data
COPY ./scripts /code/scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
