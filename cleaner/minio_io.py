import os
import io
import pandas as pd
from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_USER = os.getenv("MINIO_USER")
MINIO_PASS = os.getenv("MINIO_PASS")
MINIO_SSL = os.getenv("MINIO_SSL", "false").strip().lower() == "true"
XFORM_BUCKET = os.getenv("XFORM_BUCKET", "transform-data")

def minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=MINIO_SSL
    )

def download_csv_from_minio(corr_id: str) -> pd.DataFrame:
    cli = minio_client()
    key = f"crash/corr={corr_id}/merged.csv"
    
    resp = None
    try:
        resp = cli.get_object(XFORM_BUCKET, key)
        data = resp.read()
    finally:
        if resp:
            try:
                resp.close()
                resp.release_conn()
            except:
                pass
    
    df = pd.read_csv(io.BytesIO(data))
    return df

