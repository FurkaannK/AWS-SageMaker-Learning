import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# AWS account ID
account_id = sagemaker_session.boto_session.client(
    'sts'
).get_caller_identity()['Account']

# IAM role
SAGEMAKER_ROLE = (
    f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
)

# Get latest completed training job
training_jobs = sagemaker_session.sagemaker_client.list_training_jobs(
    SortBy='CreationTime',
    SortOrder='Descending',
    StatusEquals='Completed',
    NameContains='sklearn-modeltrainer'
)

TRAINING_JOB_NAME = training_jobs['TrainingJobSummaries'][0]['TrainingJobName']

# Endpoint name (must be unique)
ENDPOINT_NAME = "california-housing-realtime-endpoint"

try:
    # Get training job details
    training_job_details = sagemaker_session.describe_training_job(
        TrainingJobName=TRAINING_JOB_NAME
    )

    model_data = training_job_details['ModelArtifacts']['S3ModelArtifacts']

    # Create model object
    model = SKLearnModel(
        model_data=model_data,
        role=SAGEMAKER_ROLE,
        entry_point='entry_point.py',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )

    # REAL-TIME DEPLOYMENT (IMPORTANT CHANGE)
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
        wait=False
    )

    # Check endpoint status
    endpoint_description = sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=ENDPOINT_NAME
    )

    status = endpoint_description["EndpointStatus"]

    print(f"Endpoint status: {status}")

except Exception as e:
    print(f"Error: {e}")