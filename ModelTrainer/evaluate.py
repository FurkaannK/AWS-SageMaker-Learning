import tarfile
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import sagemaker
from sagemaker.s3 import S3Downloader

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# List most recent completed training jobs
training_jobs = sagemaker_session.sagemaker_client.list_training_jobs(
    SortBy='CreationTime',
    SortOrder='Descending',
    StatusEquals='Completed',
    NameContains='sklearn-modeltrainer'
)

# Extract latest training job name
TRAINING_JOB_NAME = training_jobs['TrainingJobSummaries'][0]['TrainingJobName']

# Path to test data
TEST_DATA_FILE = "data/california_housing_test.csv"

try:
    # Get training job details
    training_job_details = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=TRAINING_JOB_NAME
    )

    # Extract model artifact S3 URI
    model_s3_uri = training_job_details['ModelArtifacts']['S3ModelArtifacts']

    print(f"Downloading model from: {model_s3_uri}")

    # Download model artifact
    S3Downloader.download(model_s3_uri, ".")

    # Extract tar.gz model
    with tarfile.open("model.tar.gz", "r:gz") as tar:
        tar.extractall(".")

    # Load trained model
    model = joblib.load("model.joblib")

    # Load test dataset
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_test = test_df.drop("MedHouseVal", axis=1)
    y_test = test_df["MedHouseVal"]

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Results
    print(f"R² Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")

except Exception as e:
    print(f"Error: {e}")