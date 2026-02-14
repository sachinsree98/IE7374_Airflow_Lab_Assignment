# ML Training Pipeline with Apache Airflow

An automated Airflow pipeline that periodically trains and evaluates a Random Forest model on the California Housing dataset, fully containerized with Docker.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/sachinsree98/IE7374_Airflow_Lab_Assignment
cd airflow-ml-pipeline

# 2. Build the Docker image
docker compose build

# 3. Initialize the database and create admin user
docker compose up airflow-init

# 4. Start all services
docker compose up -d

# 5. Open the Airflow UI
# Go to http://localhost:8080
# Login — Username: admin | Password: admin

# 6. Enable and trigger the DAG
# Toggle "ml_training_pipeline" ON, then click the play button (▶) to run it

# 7. Verify model artifacts were created
cat ./models/latest_metrics.json
```

## Stopping and Restarting

```bash
# Stop (preserves data)
docker compose down

# Restart
docker compose up -d

# Full reset (wipes database)
docker compose down -v
```