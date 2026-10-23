FROM python:3.12-slim
RUN pip install --no-cache-dir "gymnasium[atari]" numpy matplotlib pillow fastapi "uvicorn[standard]" \
 && useradd -u 10001 -M app
WORKDIR /app
COPY main.py ./
COPY web ./web
ENV DADOS_DIR=/data MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
# Um único worker: o estado do treino fica na memória do processo.
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
