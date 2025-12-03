import os
import duckdb
import pandas as pd
import logging

DUCKDB_PATH = "/data/gold/gold.duckdb"

def upsert_to_gold(df: pd.DataFrame):
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
    
    # Use context manager to ensure connection is always closed
    with duckdb.connect(DUCKDB_PATH) as conn:
        # Check if table exists
        table_exists = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'crashes'
        """).fetchone()[0] > 0
        
        if not table_exists:
            # Create table with schema from first dataframe
            conn.execute("CREATE TABLE crashes AS SELECT * FROM df WHERE 1=0")
            logging.info(f"Created crashes table with {len(df.columns)} columns")
        else:
            # Get existing columns
            existing_cols = set(conn.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'crashes'
            """).fetchall())
            existing_cols = {col[0] for col in existing_cols}
            
            # Get new columns
            new_cols = set(df.columns)
            
            # Find missing columns
            missing_cols = new_cols - existing_cols
            
            # Add missing columns to table
            for col in missing_cols:
                # Determine column type from dataframe
                col_type = str(df[col].dtype)
                # Map pandas dtypes to DuckDB types
                if 'int' in col_type:
                    duckdb_type = 'BIGINT'
                elif 'float' in col_type:
                    duckdb_type = 'DOUBLE'
                elif 'bool' in col_type:
                    duckdb_type = 'BOOLEAN'
                else:
                    duckdb_type = 'VARCHAR'
                
                try:
                    conn.execute(f"ALTER TABLE crashes ADD COLUMN {col} {duckdb_type}")
                    logging.info(f"Added column {col} ({duckdb_type}) to crashes table")
                except Exception as e:
                    logging.warning(f"Could not add column {col}: {e}")
        
        # Create temp table with new data
        temp_table = "temp_staging"
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute(f"CREATE TEMPORARY TABLE {temp_table} AS SELECT * FROM df")
        
        # Get all columns that exist in both tables
        existing_cols = set(conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'crashes'
        """).fetchall())
        existing_cols = {col[0] for col in existing_cols}
        
        # Get columns from dataframe (temp table will have same columns)
        df_cols = set(df.columns)
        
        # Only use columns that exist in both
        common_cols = sorted(existing_cols & df_cols)
        
        if not common_cols:
            raise ValueError("No common columns between crashes table and new data")
        
        # Build column lists for INSERT
        cols_str = ", ".join(common_cols)
        
        # Insert only new records (by crash_record_id)
        conn.execute(f"""
            INSERT INTO crashes ({cols_str})
            SELECT {cols_str} FROM {temp_table}
            WHERE crash_record_id NOT IN (SELECT crash_record_id FROM crashes WHERE crash_record_id IS NOT NULL)
        """)
        
        total_rows = conn.execute("SELECT COUNT(*) FROM crashes").fetchone()[0]
    
    # Connection is automatically closed when exiting the 'with' block
    return total_rows
