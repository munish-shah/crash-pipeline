# Chicago Crash ETL Pipeline

## Overview

This project implements a complete end-to-end data engineering and machine learning pipeline for analyzing Chicago traffic crash data. The system extracts raw crash data from the Chicago Open Data Portal (Socrata API), transforms and cleans it through a medallion architecture (Bronze → Silver → Gold), and provides a machine learning model to predict crash severity.

### Problem Statement

The pipeline solves the challenge of processing large-scale crash data from multiple sources (crashes, vehicles, people) and making it available for analysis and predictive modeling. It automates the entire ETL process, from API extraction to clean, analytics-ready data stored in DuckDB, with real-time monitoring and a user-friendly Streamlit interface.

### Data Sources

- **Chicago Open Data Portal (Socrata API)**: 
  - Crashes dataset (ID: 85ca-t3if)
  - Vehicles dataset (ID: 68nd-jvt3)
  - People dataset (ID: u6pd-qa9d)

### Pipeline Flow

Data flows through the system in the following stages:

1. **Extraction (Bronze Layer)**: Raw JSON data is fetched from Socrata API, compressed, and stored in MinIO object storage
2. **Transformation (Silver Layer)**: JSON files are merged into a single CSV with proper joins and data type handling
3. **Cleaning (Gold Layer)**: Data quality rules are applied, outliers handled, and clean data is upserted into DuckDB
4. **ML/Analytics**: The Streamlit app provides model training, predictions, and exploratory data analysis

### ML Model

The machine learning model predicts crash severity type (binary classification):
- **Target Variable**: `crash_type`
- **Classes**: 
  - 0: No Injury/Drive Away
  - 1: Injury/Tow
- **Model Type**: Calibrated Random Forest Classifier
- **Performance**: PR-AUC: 0.576, ROC-AUC: 0.752, F1: 0.565
- **Key Features**: Weather conditions, lighting, speed limits, temporal patterns (hour, day of week), geographic location

### Streamlit Application

The Streamlit dashboard provides:
- Pipeline orchestration (trigger extract, transform, clean jobs)
- Data management and exploration
- Model training interface
- Real-time predictions with probability scores
- Exploratory data analysis with interactive visualizations
- Automated PDF report generation

---

## Architecture Diagram

[PLACEHOLDER: Add architecture diagram here showing Extractor → MinIO → Transformer → Cleaner → DuckDB → Streamlit, with Prometheus/Grafana monitoring]

---

## Pipeline Components

### Extractor (Go)

The extractor service pulls raw crash, vehicle, and people data from the Chicago Open Data Portal Socrata API. It handles pagination, rate limiting, and retries automatically. Data is compressed (gzip) and stored in MinIO object storage in the Bronze layer. The extractor exposes Prometheus metrics for monitoring API fetch duration, rows fetched, and job execution times.

**Key Features:**
- Supports both streaming (incremental) and backfill modes
- Automatic retry logic with exponential backoff
- Parallel fetching for vehicles and people datasets
- Metadata tracking with manifest files
- Publishes transform jobs to RabbitMQ after successful extraction

### Transformer (Python)

The transformer service merges the three datasets (crashes, vehicles, people) into a single denormalized CSV file. It handles data type conversions, null handling, and creates a unified schema. The merged CSV is stored in MinIO as the Silver layer.

**Key Features:**
- Efficient merging using Polars for large datasets
- Type safety and schema validation
- CSV sanitization (handles special characters, newlines)
- Publishes clean jobs to RabbitMQ after transformation

### Cleaner (Python)

The cleaner service applies data quality rules and business logic to create the Gold layer. It handles coordinate validation, outlier detection, feature engineering, and writes clean data to DuckDB.

**Key Features:**
- Geographic filtering (valid Chicago coordinates only)
- Outlier handling using IQR method
- Boolean and enum normalization
- Timestamp parsing and validation
- Upsert logic to prevent duplicates
- Tracks rows processed and dropped via Prometheus metrics

### Streamlit Application (Python)

The Streamlit app serves as the user interface for the entire pipeline. It provides tabs for data management, ETL job triggering, exploratory data analysis, model training, predictions, and report generation.

**Key Features:**
- Interactive dashboard with multiple pages
- Real-time pipeline status monitoring
- Model training with hyperparameter tuning
- Batch prediction interface
- Interactive visualizations (Plotly, Matplotlib)
- PDF report generation
- Exposes ML performance metrics to Prometheus

### Docker Compose

Orchestrates all services including:
- RabbitMQ: Message queue for job coordination
- MinIO: Object storage for Bronze and Silver layers
- Prometheus: Metrics collection and storage
- Grafana: Dashboard visualization
- RabbitMQ Exporter: Additional RabbitMQ metrics

### Monitoring

The pipeline includes comprehensive monitoring via Prometheus and Grafana:
- ETL Pipeline Overview Dashboard: Extract, Transform, Clean durations, success rates
- Storage & Queue Health Dashboard: RabbitMQ queue metrics, MinIO storage, DuckDB file size
- ML Model Performance Dashboard: Model accuracy, training duration, prediction latency, inference counts

---

## Screenshots and Demonstrations

[PLACEHOLDER: Add screenshots or video recording showing:
- Running extractor
- Running transformer
- Running cleaner
- Streamlit app (home page, train model page, prediction page)
- DuckDB tables (shown in CLI or notebook)
- Grafana dashboards showing metrics
- Prometheus target list (showing your services)]

---

## Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local Streamlit development)
- Go 1.22+ (for local extractor development)
- Git

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Pipeline
```

### 2. Create Environment File

Copy the sample environment file and fill in your values:

```bash
cp .env.sample .env
```

Edit `.env` with your configuration:
- RabbitMQ credentials
- MinIO credentials
- Socrata API token (optional, for higher rate limits)

### 3. Create Required Folders

```bash
mkdir -p minio-data prometheus_data grafana_data duckdb-data
```

### 4. Set Permissions (Linux/Mac)

```bash
chmod -R 755 minio-data prometheus_data grafana_data duckdb-data
sudo chown -R 472:472 grafana_data
```

### 5. Start Services

```bash
docker compose up -d
```

This will start:
- RabbitMQ (port 5672, management UI: 15672)
- MinIO (API: 9000, Console: 9001)
- Prometheus (port 9090)
- Grafana (port 3000)
- Extractor, Transformer, Cleaner services

### 6. Start Streamlit Application

In a separate terminal:

```bash
cd streamlit-app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`

---

## Running the Pipeline

### Method 1: Via Streamlit Dashboard

1. Open the Streamlit app at `http://localhost:8501`
2. Navigate to the "Data Fetcher" tab
3. Configure your extraction job (streaming or backfill mode)
4. Click "Trigger Extract Job"
5. Monitor progress in the "Data Management" tab

The pipeline will automatically:
- Extract data → Transform → Clean → Store in DuckDB

### Method 2: Via RabbitMQ (Programmatic)

Publish a job message to the `extract` queue:

```json
{
  "mode": "streaming",
  "corr_id": "2025-12-01T10-00-00Z",
  "source": "https://data.cityofchicago.org",
  "join_key": "crash_record_id",
  "primary": {
    "id": "85ca-t3if",
    "alias": "crashes",
    "select": "crash_record_id,crash_date,weather_condition,lighting_condition",
    "where": "crash_date >= '2024-01-01T00:00:00'"
  },
  "enrich": [
    {
      "id": "68nd-jvt3",
      "alias": "vehicles",
      "select": "crash_record_id,make,model,vehicle_year"
    },
    {
      "id": "u6pd-qa9d",
      "alias": "people",
      "select": "crash_record_id,person_type,age"
    }
  ],
  "storage": {
    "bucket": "raw-data",
    "prefix": "crash",
    "compress": true
  }
}
```

### Accessing Services

- **Streamlit**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (admin/admin123)
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)

---

## Project Structure

```
Pipeline/
├── extractor/               # Go extractor (bronze layer)
│   ├── main.go
│   ├── go.mod
│   ├── go.sum
│   └── Dockerfile
├── transformer/             # Python transformer (silver layer)
│   ├── transformer.py
│   ├── requirements.txt
│   └── Dockerfile
├── cleaner/                 # Python cleaner (gold layer)
│   ├── cleaner.py
│   ├── cleaning_rules.py
│   ├── duckdb_writer.py
│   ├── minio_io.py
│   ├── requirements.txt
│   └── Dockerfile
├── streamlit-app/           # Streamlit UI
│   ├── streamlit_app.py
│   ├── artifacts/           # ML model artifacts
│   │   ├── model.pkl
│   │   ├── threshold.txt
│   │   └── labels.json
│   └── requirements.txt
├── monitoring/              # Monitoring configuration
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── create_dashboards.py
├── docker-compose.yaml      # Orchestrates all services
├── .env.sample              # Environment variable template
├── .gitignore               # Git ignore rules
├── backfill.json            # Backfill job configuration
├── streaming.json           # Streaming job configuration
└── README.md                # This file
```

---

## Improvements and Extra Features

### Data Quality Enhancements
- Comprehensive coordinate validation (Chicago boundaries)
- Outlier detection using IQR method for numeric columns
- Boolean and enum normalization for consistency
- Timestamp parsing with timezone handling

### Monitoring and Observability
- Prometheus metrics exposed by all services
- Three Grafana dashboards (ETL, Storage/Queue, ML Performance)
- Custom metrics: Pipeline Success Rate, MinIO Free Space Percentage
- Real-time monitoring of job durations, error rates, and throughput

### ML Model Features
- Calibrated classifier for reliable probability estimates
- Feature engineering (interaction terms, flags)
- Class imbalance handling with balanced class weights
- Decision threshold optimization for F1 score

### Streamlit Dashboard
- Multi-page interface with navigation
- Interactive visualizations (Plotly charts)
- Real-time pipeline status updates
- Automated PDF report generation
- Batch prediction interface with ground truth comparison

### Pipeline Reliability
- Automatic retry logic with exponential backoff
- Health checks for all Docker services
- Message queue persistence (RabbitMQ)
- Data lineage tracking with manifest files

---

## Lessons Learned and Challenges

### Challenges Encountered

1. **Data Volume and Performance**: Processing large datasets (100K+ rows) required optimization. Used Polars instead of Pandas for faster transformations and efficient memory usage.

2. **Class Imbalance**: The crash severity dataset was imbalanced (27% positive class). Addressed with balanced class weights and PR-AUC as the primary metric.

3. **Coordinate Validation**: Many crash records had invalid or out-of-bounds coordinates. Implemented strict geographic filtering to ensure data quality.

4. **Streamlit Hot Reloading**: Prometheus metrics registration caused duplicate errors on Streamlit reruns. Solved with a custom metric registry check.

5. **Docker Volume Permissions**: Grafana required specific UID (472) for data persistence. Fixed with proper chown commands.

### What I Learned

- End-to-end pipeline design with proper separation of concerns (Bronze/Silver/Gold)
- Message queue patterns for decoupled service communication
- Prometheus metrics instrumentation and Grafana dashboard creation
- Handling imbalanced classification problems in ML
- Docker Compose orchestration for multi-service applications
- Data quality validation and outlier detection techniques

### Future Improvements

If given more time, I would:
- Add data validation schemas (e.g., Great Expectations)
- Implement automated data quality reports
- Add more sophisticated feature engineering (temporal features, location clustering)
- Implement model versioning and A/B testing
- Add alerting rules for pipeline failures
- Create CI/CD pipeline for automated testing and deployment
- Add support for real-time streaming predictions
- Implement model retraining automation based on data drift detection

---

## License

[Add your license here if applicable]

---

## Author

Munish Shah

