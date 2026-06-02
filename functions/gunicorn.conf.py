import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
graceful_timeout = 30
keepalive = 5
