import os
import duckdb
import pandas as pd

DUCKDB_PATH = "/data/gold/gold.duckdb"

def upsert_to_gold(df: pd.DataFrame):
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
    
    # Use context manager to ensure connection is always closed
    with duckdb.connect(DUCKDB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS crashes AS SELECT * FROM df WHERE 1=0")
        
        temp_table = "temp_staging"
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute(f"CREATE TEMPORARY TABLE {temp_table} AS SELECT * FROM df")
        
        conn.execute(f"""
            INSERT INTO crashes 
            SELECT * FROM {temp_table}
            WHERE crash_record_id NOT IN (SELECT crash_record_id FROM crashes)
        """)
        
        total_rows = conn.execute("SELECT COUNT(*) FROM crashes").fetchone()[0]
    
    # Connection is automatically closed when exiting the 'with' block
    return total_rows
