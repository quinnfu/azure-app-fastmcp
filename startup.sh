#!/bin/bash

# Try to activate virtual environment if it exists (Oryx standard path)
if [ -d "antenv" ]; then
    source antenv/bin/activate
elif [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
fi

# Run the application
# Using gunicorn with uvicorn workers for production
# -w 4: 4 workers
# -k uvicorn.workers.UvicornWorker: use uvicorn
# --timeout 600: 10 minute timeout (good for long LLM responses)
# --access-logfile - --error-logfile -: send logs to stdout/stderr (Azure Log Stream)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:fastmcp_asgi_app --bind=0.0.0.0:8000 --timeout 600 --access-logfile - --error-logfile -
