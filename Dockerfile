FROM python:3.12-slim

# libpq-dev/gcc for psycopg's build; rasterio/shapely ship self-contained
# manylinux wheels (bundled GDAL/GEOS) so no separate system GDAL install
# is needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY geocadastra/ geocadastra/
COPY migrations/ migrations/
COPY scripts/render_bootstrap.py scripts/render_bootstrap.py

EXPOSE 10000
# render_bootstrap runs every start (idempotent -- see its own docstring),
# so the web service's default CMD is self-contained: no separate
# pre-deploy step to configure. The worker service overrides this CMD
# (dockerCommand: celery ...) and does not need the bootstrap step at all.
# $PORT: Render injects this at runtime -- shell-form CMD expands it;
# ${PORT:-10000} keeps `docker run` usable locally without Render's env.
CMD python scripts/render_bootstrap.py && python -m uvicorn geocadastra.api.main:app --host 0.0.0.0 --port ${PORT:-10000}
