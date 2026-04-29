FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


# install dependencies (add any additional dependencies your project requires)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow transformers


WORKDIR /app
COPY models/ /app/models/
COPY weights/ /app/weights/
COPY predict_peak.py /app/predict_peak.py
COPY predict_efficiency.py /app/predict_efficiency.py


ENTRYPOINT ["python"]