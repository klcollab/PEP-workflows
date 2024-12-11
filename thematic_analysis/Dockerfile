FROM python:3.12.3-bookworm
COPY *.py requirements.txt /app/
RUN python -m pip install -Ur /app/requirements.txt
COPY run.sh /app/
