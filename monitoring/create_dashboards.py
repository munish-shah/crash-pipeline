#!/usr/bin/env python3
"""
Create Grafana dashboards for the crash pipeline monitoring
"""
import json
import requests
import sys
import subprocess
import os

GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"

def get_grafana_password():
    """Get Grafana password from docker container"""
    try:
        result = subprocess.run(
            ['docker', 'compose', 'exec', '-T', 'grafana', 'printenv', 'GF_SECURITY_ADMIN_PASSWORD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)) + '/..'
        )
        password = result.stdout.strip()
        if password:
            return password
    except:
        pass
    # Fallback to default
    return "admin"

GRAFANA_PASS = get_grafana_password()

def get_dashboard_by_title(title):
    """Get existing dashboard by title"""
    url = f"{GRAFANA_URL}/api/search"
    auth = (GRAFANA_USER, GRAFANA_PASS)
    params = {"query": title}
    
    response = requests.get(url, auth=auth, params=params, timeout=10)
    if response.status_code == 200:
        dashboards = response.json()
        for dash in dashboards:
            if dash.get('title') == title:
                return dash.get('uid')
    return None

def get_dashboard_json(uid):
    """Get full dashboard JSON by UID"""
    url = f"{GRAFANA_URL}/api/dashboards/uid/{uid}"
    auth = (GRAFANA_USER, GRAFANA_PASS)
    
    response = requests.get(url, auth=auth, timeout=10)
    if response.status_code == 200:
        return response.json().get('dashboard')
    return None

def create_or_update_dashboard(dashboard_json):
    """Create or update a dashboard in Grafana"""
    title = dashboard_json['dashboard']['title']
    url = f"{GRAFANA_URL}/api/dashboards/db"
    headers = {"Content-Type": "application/json"}
    auth = (GRAFANA_USER, GRAFANA_PASS)
    
    # Check if dashboard exists
    existing_uid = get_dashboard_by_title(title)
    if existing_uid:
        # Get existing dashboard to preserve UID and version
        existing = get_dashboard_json(existing_uid)
        if existing:
            dashboard_json['dashboard']['uid'] = existing_uid
            dashboard_json['dashboard']['version'] = existing.get('version', 0) + 1
            dashboard_json['dashboard']['id'] = existing.get('id')
            print(f"📝 Updating existing dashboard '{title}' (uid: {existing_uid})")
        else:
            print(f"⚠️  Found dashboard but couldn't fetch details, creating new version")
    
    response = requests.post(url, json=dashboard_json, headers=headers, auth=auth, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Dashboard '{title}' {'updated' if existing_uid else 'created'} successfully")
        print(f"   URL: {GRAFANA_URL}{result.get('url', '')}")
        return True
    else:
        print(f"❌ Failed to {'update' if existing_uid else 'create'} dashboard: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        return False

# Dashboard 1: ETL Overview
dashboard1 = {
    "dashboard": {
        "title": "ETL Pipeline Overview",
        "tags": ["etl", "pipeline"],
        "timezone": "browser",
        "schemaVersion": 16,
        "version": 0,
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "Extract Duration (avg)",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                "targets": [{
                    "expr": "rate(extractor_duration_seconds_sum[5m]) / rate(extractor_duration_seconds_count[5m])",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "decimals": 3
                    }
                }
            },
            {
                "id": 2,
                "title": "Transform Duration (avg)",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                "targets": [{
                    "expr": "rate(transformer_duration_seconds_sum[5m]) / rate(transformer_duration_seconds_count[5m])",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "decimals": 3
                    }
                }
            },
            {
                "id": 3,
                "title": "Clean Duration (avg)",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                "targets": [{
                    "expr": "rate(cleaner_duration_seconds_sum[5m]) / rate(cleaner_duration_seconds_count[5m])",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "decimals": 3
                    }
                }
            },
            {
                "id": 4,
                "title": "Rows Processed by Stage",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "extractor_rows_fetched_total",
                        "legendFormat": "Extracted",
                        "refId": "A"
                    },
                    {
                        "expr": "transformer_rows_processed_total",
                        "legendFormat": "Transformed",
                        "refId": "B"
                    },
                    {
                        "expr": "cleaner_rows_processed_total",
                        "legendFormat": "Cleaned",
                        "refId": "C"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short"
                    }
                }
            },
            {
                "id": 5,
                "title": "Error Counts by Service",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    {
                        "expr": "extractor_errors_total",
                        "legendFormat": "Extractor Errors",
                        "refId": "A"
                    },
                    {
                        "expr": "transformer_errors_total",
                        "legendFormat": "Transformer Errors",
                        "refId": "B"
                    },
                    {
                        "expr": "cleaner_errors_total",
                        "legendFormat": "Cleaner Errors",
                        "refId": "C"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short"
                    }
                }
            },
            {
                "id": 6,
                "title": "Service Uptime",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [
                    {
                        "expr": "extractor_uptime_seconds",
                        "legendFormat": "Extractor",
                        "refId": "A"
                    },
                    {
                        "expr": "transformer_uptime_seconds",
                        "legendFormat": "Transformer",
                        "refId": "B"
                    },
                    {
                        "expr": "cleaner_uptime_seconds",
                        "legendFormat": "Cleaner",
                        "refId": "C"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s"
                    }
                }
            },
            {
                "id": 7,
                "title": "Custom: Pipeline Success Rate",
                "type": "stat",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                "targets": [{
                    "expr": "(sum(extractor_runs_total) + sum(transformer_runs_total) + sum(cleaner_runs_total) - sum(extractor_errors_total) - sum(transformer_errors_total) - sum(cleaner_errors_total)) / (sum(extractor_runs_total) + sum(transformer_runs_total) + sum(cleaner_runs_total)) * 100",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "decimals": 1
                    }
                },
                "description": "Custom metric: Overall pipeline health - percentage of successful runs across all services"
            }
        ]
    },
    "overwrite": True
}

# Dashboard 2: Storage & Queue Health
dashboard2 = {
    "dashboard": {
        "title": "Storage & Queue Health",
        "tags": ["storage", "queue", "rabbitmq", "minio"],
        "timezone": "browser",
        "schemaVersion": 16,
        "version": 0,
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "RabbitMQ Queue Size (Messages Ready)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [{
                    "expr": "rabbitmq_queue_messages_ready",
                    "legendFormat": "{{queue}}",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short",
                        "nullValueMode": "null"
                    }
                },
                "description": "Number of messages waiting in each queue"
            },
            {
                "id": 2,
                "title": "RabbitMQ Messages Published vs Consumed",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "rate(rabbitmq_queue_messages_published_total[5m])",
                        "legendFormat": "Published ({{queue}})",
                        "refId": "A"
                    },
                    {
                        "expr": "rate(rabbitmq_queue_messages_delivered_total[5m])",
                        "legendFormat": "Consumed ({{queue}})",
                        "refId": "B"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "ops",
                        "nullValueMode": "null"
                    }
                },
                "description": "Message throughput: published vs consumed per second"
            },
            {
                "id": 3,
                "title": "MinIO Total Object Count",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [{
                    "expr": "minio_cluster_usage_object_total",
                    "legendFormat": "Total Objects",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short",
                        "nullValueMode": "null"
                    }
                },
                "description": "Total number of objects stored in MinIO cluster (across all buckets)"
            },
            {
                "id": 4,
                "title": "MinIO Total Storage Used",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [{
                    "expr": "minio_cluster_usage_total_bytes",
                    "legendFormat": "Total Bytes",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "bytes",
                        "nullValueMode": "null"
                    }
                },
                "description": "Total size of all objects in MinIO cluster (across all buckets)"
            },
            {
                "id": 5,
                "title": "MinIO Cluster Bucket Count",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 16},
                "targets": [{
                    "expr": "minio_cluster_bucket_total",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short"
                    }
                }
            },
            {
                "id": 6,
                "title": "MinIO Capacity Usage",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 16},
                "targets": [{
                    "expr": "(1 - (minio_cluster_capacity_raw_free_bytes / minio_cluster_capacity_raw_total_bytes)) * 100",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100
                    }
                },
                "description": "Percentage of MinIO storage capacity used"
            },
            {
                "id": 7,
                "title": "Rows in Gold Database (DuckDB)",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 24},
                "targets": [{
                    "expr": "cleaner_rows_processed_total",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short",
                        "noValue": "0"
                    }
                },
                "description": "Total rows processed and written to DuckDB Gold table (counter - shows rows processed since service start)"
            },
            {
                "id": 10,
                "title": "DuckDB File Size",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 24},
                "targets": [{
                    "expr": "cleaner_duckdb_file_size_bytes",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "bytes"
                    }
                },
                "description": "Size of DuckDB Gold database file (using Cleaner metric)"
            },
            {
                "id": 8,
                "title": "Custom: MinIO Free Space Percentage",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                "targets": [{
                    "expr": "(minio_cluster_capacity_raw_free_bytes / minio_cluster_capacity_raw_total_bytes) * 100",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"value": 0, "color": "red"},
                                {"value": 10, "color": "yellow"},
                                {"value": 20, "color": "green"}
                            ]
                        }
                    }
                },
                "description": "Custom metric: Percentage of free storage space remaining in MinIO (alert if < 10%)"
            }
        ]
    },
    "overwrite": True
}

# Dashboard 3: ML Performance
dashboard3 = {
    "dashboard": {
        "title": "ML Model Performance",
        "tags": ["ml", "model", "streamlit"],
        "timezone": "browser",
        "schemaVersion": 16,
        "version": 0,
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "Model Accuracy",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                "targets": [{
                    "expr": "streamlit_model_accuracy",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percentunit",
                        "decimals": 3,
                        "min": 0,
                        "max": 1
                    }
                },
                "description": "Model accuracy (or main metric you track)"
            },
            {
                "id": 2,
                "title": "Training Duration",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                "targets": [{
                    "expr": "streamlit_training_duration_seconds",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "decimals": 1
                    }
                },
                "description": "Duration of model training in seconds"
            },
            {
                "id": 3,
                "title": "Inference Count",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                "targets": [{
                    "expr": "streamlit_predictions_total",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short"
                    }
                },
                "description": "Total number of predictions made (inference count)"
            },
            {
                "id": 4,
                "title": "Last-Trained Timestamp",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                "targets": [{
                    "expr": "streamlit_last_trained_timestamp",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "dateTimeFromNow"
                    }
                },
                "description": "Unix timestamp of last model training"
            },
            {
                "id": 5,
                "title": "Custom: Prediction Latency (p95)",
                "type": "stat",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [{
                    "expr": "histogram_quantile(0.95, sum(rate(streamlit_prediction_duration_seconds_bucket[5m])) by (le))",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "decimals": 3,
                        "nullValueMode": "null"
                    }
                },
                "description": "Custom metric: 95th percentile prediction latency"
            }
        ]
    },
    "overwrite": True
}

if __name__ == "__main__":
    print("Creating/Updating Grafana dashboards...")
    print("=" * 50)
    print(f"Grafana URL: {GRAFANA_URL}")
    print(f"Username: {GRAFANA_USER}")
    print("=" * 50)
    
    success = True
    success &= create_or_update_dashboard(dashboard1)
    success &= create_or_update_dashboard(dashboard2)
    success &= create_or_update_dashboard(dashboard3)
    
    if success:
        print("=" * 50)
        print("✅ All dashboards created/updated successfully!")
        print(f"\nAccess Grafana at: {GRAFANA_URL}")
        sys.exit(0)
    else:
        print("=" * 50)
        print("❌ Some dashboards failed to create/update")
        print("\n💡 If authentication failed, check:")
        print("   1. Grafana container is running")
        print("   2. Password in docker-compose.yaml matches")
        print("   3. Try accessing Grafana UI to verify credentials")
        sys.exit(1)


