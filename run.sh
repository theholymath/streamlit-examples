#!/bin/sh
docker run --rm -ti \
    -v $(pwd):/app \
    -p 8555:8501 \
    -u root \
    -w /app \
    streamlit streamlit run \
    --browser.serverAddress 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False \
    $1
