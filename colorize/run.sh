#!/bin/sh
streamlit run \
    --browser.serverAddress 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False \
    app.py
