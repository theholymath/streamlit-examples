#!/bin/sh
docker run --rm -ti \
    -v $(pwd):/app \
    -p 8501:8501 \
    -p 8502:8502 \
    -u root \
    streamlit streamlit run \
    --browser.serverAddress 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False \
    $1
