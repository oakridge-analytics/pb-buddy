import multiprocessing

# Server socket settings
bind = "0.0.0.0:5000"

# Timeout settings
timeout = 120  # Increase worker timeout to 120 seconds
graceful_timeout = 120  # Time for graceful worker shutdown
keepalive = 5  # How long to wait for requests on a Keep-Alive connection

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = "info"

# Process naming
proc_name = "bike.broker"

# SSL config if needed
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50  # Adds randomness to max_requests

# Limit the size of incoming request bodies
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Worker processes
workers = 2  # Start with single worker
worker_class = "gthread"  # Use threads
threads = 4  # Number of threads per worker
