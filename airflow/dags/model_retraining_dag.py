from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import logging
import pandas as pd

# Default arguments
default_args = {
    'owner': 'filmfusion-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
dag = DAG(
    'filmfusion_model_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline for FilmFusion',
    schedule_interval=timedelta(days=7),  # Weekly retraining
    max_active_runs=1,
    tags=['filmfusion', 'ml', 'retraining']
)

def check_data_quality(**context):
    """Check data quality before retraining"""
    from src.pipelines.evaluation_pipeline import DataQualityChecker
    
    checker = DataQualityChecker()
    
    # Load latest data
    ratings_df = pd.read_csv('/app/data/processed/ratings_cleaned.csv')
    movies_df = pd.read_csv('/app/data/enriched/movies_enriched.csv')
    
    # Run quality checks
    quality_report = checker.run_quality_checks(ratings_df, movies_df)
    
    # Log results
    logging.info(f"Data quality report: {quality_report}")
    
    # Fail if data quality is poor
    if quality_report['overall_score'] < 0.8:
        raise ValueError(f"Data quality too low: {quality_report['overall_score']}")
    
    return quality_report

def detect_model_drift(**context):
    """Detect if models need retraining due to performance drift"""
    from src.models.retraining.drift_detector import DriftDetector
    
    drift_detector = DriftDetector()
    
    # Load current model performance
    current_performance = drift_detector.load_current_performance()
    
    # Load recent evaluation metrics
    recent_performance = drift_detector.evaluate_recent_performance()
    
    # Check for drift
    drift_detected = drift_detector.detect_drift(current_performance, recent_performance)
    
    logging.info(f"Drift detection result: {drift_detected}")
    
    # Store result for downstream tasks
    context['task_instance'].xcom_push(key='drift_detected', value=drift_detected)
    
    return drift_detected

def retrain_models(**context):
    """Retrain all models"""
    from src.pipelines.training_pipeline import ModelTrainingPipeline
    
    # Check if retraining is needed
    drift_detected = context['task_instance'].xcom_pull(
        task_ids='detect_drift', key='drift_detected'
    )
    
    if not drift_detected:
        logging.info("No drift detected, skipping retraining")
        return "skipped"
    
    # Initialize training pipeline
    training_pipeline = ModelTrainingPipeline()
    
    # Load latest data
    ratings_df = pd.read_csv('/app/data/processed/ratings_cleaned.csv')
    movies_df = pd.read_csv('/app/data/enriched/movies_enriched.csv')
    
    # Run training pipeline
    training_results = training_pipeline.run_full_training(ratings_df, movies_df)
    
    logging.info(f"Training completed: {training_results}")
    
    return training_results

def evaluate_new_models(**context):
    """Evaluate newly trained models"""
    from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline
    
    evaluator = ModelEvaluationPipeline()
    
    # Load test data
    test_df = pd.read_csv('/app/data/processed/test_ratings.csv')
    
    # Evaluate all models
    evaluation_results = evaluator.evaluate_all_models(test_df)
    
    logging.info(f"Evaluation results: {evaluation_results}")
    
    # Check if new models perform better
    improvement_threshold = 0.02  # 2% improvement required
    best_model = max(evaluation_results.keys(), 
                    key=lambda x: evaluation_results[x]['f1_at_k'])
    
    context['task_instance'].xcom_push(key='evaluation_results', value=evaluation_results)
    context['task_instance'].xcom_push(key='best_model', value=best_model)
    
    return evaluation_results

def deploy_models(**context):
    """Deploy models if they perform better"""
    from src.models.retraining.model_deployer import ModelDeployer
    
    evaluation_results = context['task_instance'].xcom_pull(
        task_ids='evaluate_models', key='evaluation_results'
    )
    
    best_model = context['task_instance'].xcom_pull(
        task_ids='evaluate_models', key='best_model'
    )
    
    deployer = ModelDeployer()
    
    # Deploy best performing model
    deployment_result = deployer.deploy_model(best_model, evaluation_results)
    
    logging.info(f"Deployment result: {deployment_result}")
    
    return deployment_result

def send_notification(**context):
    """Send notification about retraining results"""
    
    # Get results from previous tasks
    training_results = context['task_instance'].xcom_pull(task_ids='retrain_models')
    evaluation_results = context['task_instance'].xcom_pull(
        task_ids='evaluate_models', key='evaluation_results'
    )
    deployment_result = context['task_instance'].xcom_pull(task_ids='deploy_models')
    
    # Create notification message
    message = f"""
    FilmFusion Model Retraining Completed
    
    Training: {training_results}
    Evaluation: {evaluation_results}
    Deployment: {deployment_result}
    
    Timestamp: {datetime.now()}
    """
    
    logging.info(f"Retraining pipeline completed: {message}")
    
    # In production, send email/Slack notification
    return message

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

detect_drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_model_drift,
    dag=dag
)

retrain_task = PythonOperator(
    task_id='retrain_models',
    python_callable=retrain_models,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_new_models,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_models',
    python_callable=deploy_models,
    dag=dag
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

# Define task dependencies
check_data_task >> detect_drift_task >> retrain_task >> evaluate_task >> deploy_task >> notification_task
