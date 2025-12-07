import os
import logging
import pika
import json
import time
import socket
import random
import threading
from minio_io import download_csv_from_minio
from cleaning_rules import clean_dataframe
from duckdb_writer import upsert_to_gold
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="[cleaner] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)
intentional bug
# ---------------------------------
# Prometheus Metrics
# ---------------------------------
# Service uptime
cleaner_uptime = Gauge('cleaner_uptime_seconds', 'Service uptime in seconds')
cleaner_start_time = time.time()

# Run counters
cleaner_runs_total = Counter('cleaner_runs_total', 'Total number of clean jobs executed')
cleaner_errors_total = Counter('cleaner_errors_total', 'Total number of clean job errors')

# Duration metrics
cleaner_duration_seconds = Histogram('cleaner_duration_seconds', 'Duration of clean job execution', buckets=[1, 5, 10, 30, 60, 120, 300])

# Row processing metrics
cleaner_rows_processed = Counter('cleaner_rows_processed_total', 'Total number of rows processed')
cleaner_rows_dropped = Counter('cleaner_rows_dropped_total', 'Total number of rows dropped during cleaning')

# Custom metric: DuckDB write duration
cleaner_duckdb_write_duration_seconds = Histogram('cleaner_duckdb_write_duration_seconds', 'Duration of DuckDB write operation', buckets=[0.1, 0.5, 1, 2, 5, 10, 30])

# Custom metric: DuckDB file size
cleaner_duckdb_file_size_bytes = Gauge('cleaner_duckdb_file_size_bytes', 'Size of DuckDB Gold database file in bytes')

def update_uptime():
    """Update uptime gauge periodically"""
    while True:
        cleaner_uptime.set(time.time() - cleaner_start_time)
        # Also update DuckDB file size periodically
        duckdb_path = "/data/gold/gold.duckdb"
        if os.path.exists(duckdb_path):
            try:
                file_size = os.path.getsize(duckdb_path)
                cleaner_duckdb_file_size_bytes.set(file_size)
            except Exception:
                pass
        time.sleep(10)  # Update every 10 seconds

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
CLEAN_QUEUE = os.getenv("CLEAN_QUEUE", "clean")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_USER = os.getenv("MINIO_USER")
MINIO_PASS = os.getenv("MINIO_PASS")
MINIO_SSL = os.getenv("MINIO_SSL", "false").strip().lower() == "true"
XFORM_BUCKET = os.getenv("XFORM_BUCKET")

def process_clean_job(msg: dict):
    job_start = time.time()
    cleaner_runs_total.inc()
    
    try:
        corr_id = msg.get("corr_id")
        
        logging.info(f"Received clean job corr={corr_id}")
        
        df = download_csv_from_minio(corr_id)
        initial_rows = len(df)
        logging.info(f"Downloaded CSV: {initial_rows} rows, {len(df.columns)} cols")
        
        cleaned_df = clean_dataframe(df)
        final_rows = len(cleaned_df)
        rows_dropped = initial_rows - final_rows
        
        # Record rows dropped (always record, even if 0, to show metric exists)
        if rows_dropped > 0:
            cleaner_rows_dropped.inc(rows_dropped)
            logging.info(f"Dropped {rows_dropped} rows during cleaning")
        
        logging.info(f"After cleaning: {final_rows} rows, {len(cleaned_df.columns)} cols")
        
        # Measure DuckDB write duration
        write_start = time.time()
        upsert_to_gold(cleaned_df)
        write_duration = time.time() - write_start
        cleaner_duckdb_write_duration_seconds.observe(write_duration)
        
        # Update DuckDB file size metric
        duckdb_path = "/data/gold/gold.duckdb"
        if os.path.exists(duckdb_path):
            file_size = os.path.getsize(duckdb_path)
            cleaner_duckdb_file_size_bytes.set(file_size)
        
        # Record rows processed
        cleaner_rows_processed.inc(final_rows)
        
        logging.info(f"Upserted {final_rows} rows to gold.crashes")
        
        # Record successful duration
        cleaner_duration_seconds.observe(time.time() - job_start)
    
    except Exception as e:
        cleaner_errors_total.inc()
        cleaner_duration_seconds.observe(time.time() - job_start)
        raise

def wait_for_port(host: str, port: int, tries: int = 60, delay: float = 1.0):
    for _ in range(tries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False

def start_consumer():
    from pika.exceptions import AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError
    
    params = pika.URLParameters(RABBITMQ_URL)
    host = params.host or "rabbitmq"
    port = params.port or 5672
    
    if not wait_for_port(host, port, tries=60, delay=1.0):
        raise SystemExit(f"[cleaner] RabbitMQ not reachable at {host}:{port}")
    
    max_tries = 60
    base_delay = 1.5
    conn = None
    
    for i in range(1, max_tries + 1):
        try:
            conn = pika.BlockingConnection(params)
            break
        except (AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError) as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBITMQ_URL}")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/{max_tries}): {e.__class__.__name__}")
            time.sleep(base_delay + random.random())
    
    if conn is None or not conn.is_open:
        raise SystemExit("[cleaner] Could not connect to RabbitMQ")
    
    ch = conn.channel()
    ch.queue_declare(queue=CLEAN_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)
    
    def on_msg(chx, method, props, body):
        try:
            msg = json.loads(body.decode("utf-8"))
            process_clean_job(msg)
            chx.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    logging.info(f"Up. Waiting for jobs on queue '{CLEAN_QUEUE}'")
    ch.basic_consume(queue=CLEAN_QUEUE, on_message_callback=on_msg)
    
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        try: ch.stop_consuming()
        except Exception: pass
        try: conn.close()
        except Exception: pass

if __name__ == "__main__":
    # Initialize DuckDB file size metric on startup
    duckdb_path = "/data/gold/gold.duckdb"
    if os.path.exists(duckdb_path):
        try:
            file_size = os.path.getsize(duckdb_path)
            cleaner_duckdb_file_size_bytes.set(file_size)
            logging.info(f"Initialized DuckDB file size metric: {file_size} bytes")
        except Exception as e:
            logging.warning(f"Could not initialize DuckDB file size: {e}")
    
    # Initialize rows processed counter with actual DB count
    try:
        import duckdb
        with duckdb.connect(duckdb_path) as conn:
            result = conn.execute("SELECT COUNT(*) FROM crashes").fetchone()
            if result and result[0] > 0:
                actual_rows = result[0]
                # Set the counter to the actual count (counters can be set to a value)
                cleaner_rows_processed._value._value = actual_rows
                logging.info(f"Initialized rows processed counter: {actual_rows} rows")
    except Exception as e:
        logging.warning(f"Could not initialize rows counter from DB: {e}")
    
    # Start Prometheus metrics server
    start_http_server(8080)
    logging.info("Prometheus metrics server started on port 8080")
    
    # Start uptime updater thread
    uptime_thread = threading.Thread(target=update_uptime, daemon=True)
    uptime_thread.start()
    
    start_consumer()

