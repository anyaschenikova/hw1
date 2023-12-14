FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir numba
RUN pip install --no-cache-dir tqdm

CMD ["python3", "hw1_0_1.py"]