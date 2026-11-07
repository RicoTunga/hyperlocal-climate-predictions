# Dashboard image: the forecast-replay web app with sign-in. Build it on a machine that has
# app/data/ (python -m app.build_data) and the committed reports/. See deploy/README.md.
#
#   docker build -t hyperlocal-dashboard .
#   docker run -p 8765:8765 -v hyperlocal-auth:/data hyperlocal-dashboard
#
# The container listens on 0.0.0.0:8765 and expects HTTPS to be provided in front of it (the hosting
# platform or a reverse proxy such as Caddy). Users and sessions live in the /data volume.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HL_HOST=0.0.0.0 \
    HL_PORT=8765 \
    HL_AUTH_DB=/data/auth/users.db \
    HL_SECURE_COOKIES=1

WORKDIR /srv
RUN useradd --create-home --uid 10001 app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# only what the web app needs: code, the dashboard data and the committed result tables
COPY app/__init__.py app/components.py app/
COPY app/data app/data
COPY web web
COPY reports/test reports/test
COPY reports/late_blight reports/late_blight

RUN mkdir -p /data/auth && chown -R app:app /data
USER app
VOLUME /data
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz')"

# --proxy-headers: trust X-Forwarded-Proto/For from the HTTPS proxy in front (so cookies are marked
# Secure and client IPs are right). Only expose the container through that proxy.
CMD ["uvicorn", "web.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765", "--proxy-headers", "--forwarded-allow-ips", "*"]
