from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Import your ETL functions
import sys
import os
sys.path.append('/app')
from src.data_processing.etl_pipeline import ETLPipeline

default_args = {
    'owner': 'filmfusion',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'daily_data_refresh',
    default_args=default_args,
    description='Daily data refresh for FilmFusion',
    schedule_interval='@daily',
    catchup=False
)

def run_etl_pipeline():
    """Run the ETL pipeline"""
    etl = ETLPipeline()
    return etl.run_full_pipeline()

# Tasks
etl_task = PythonOperator(
    task_id='run_etl_pipeline',
    python_callable=run_etl_pipeline,
    dag=dag
)

# Data quality check
data_quality_check = BashOperator(
    task_id='data_quality_check',
    bash_command='python /app/scripts/data_quality_check.py',
    dag=dag
)

etl_task >> data_quality_check
