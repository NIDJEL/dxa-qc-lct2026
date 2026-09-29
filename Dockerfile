FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924
WORKDIR /app
RUN mkdir -p /app/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-inference.txt /app/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-inference.txt
COPY src /app/src
COPY scripts/run_dxa_qc_service.py scripts/run_dxa_qc_web.py scripts/benchmark_submission.py /app/scripts/
ENV PYTHONPATH=/app/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENV DXA_QC_ROOT=/app DXA_QC_HOST=0.0.0.0 DXA_QC_JOBS_DIR=/tmp/dxa_qc_jobs
RUN useradd --system --uid 10001 --create-home dxa && mkdir -p /tmp/dxa_qc_jobs && chown -R dxa:dxa /tmp/dxa_qc_jobs /app
USER dxa
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
ENTRYPOINT ["python", "scripts/run_dxa_qc_web.py"]
