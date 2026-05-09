import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get the default SageMaker bucket name
default_bucket = sagemaker_session.default_bucket()

# Get account ID
account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']

# S3 data URI where the training data is stored
S3_TRAIN_DATA_URI = f"s3://{default_bucket}/datasets/california_housing_train.csv"

# Type of instance to use for training
INSTANCE_TYPE = "ml.m5.large"
# Number of instances to use for training
INSTANCE_COUNT = 1
# SageMaker execution role
SAGEMAKER_ROLE = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
# Custom output location
MODEL_OUTPUT_PATH = f"s3://{default_bucket}/models/california-housing/"

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',              # Python script containing training code
    role=SAGEMAKER_ROLE,                 # IAM role for SageMaker to access AWS resources
    instance_type=INSTANCE_TYPE,         # Type of EC2 instance for training
    instance_count=INSTANCE_COUNT,       # Number of instances to use
    framework_version='1.2-1',           # Version of scikit-learn to use
    py_version='py3',                    # Python version for the training environment
    script_mode=True,                    # Use script mode for training (vs legacy mode)
    sagemaker_session=sagemaker_session, # Session for interacting with SageMaker
    output_path=MODEL_OUTPUT_PATH        # S3 location to save model artifacts
)

try:
    # Start training job asynchronously
    sklearn_estimator.fit(
        {"train": S3_TRAIN_DATA_URI},
        wait=False
    )

    # Get training job name
    training_job_name = sklearn_estimator.latest_training_job.name

    # Print training job name
    print(f"Training Job Name: {training_job_name}")

    # Get current job status
    job_status = (
        sklearn_estimator
        .latest_training_job
        .describe()["TrainingJobStatus"]
    )

    # Print job status
    print(f"Training Job Status: {job_status}")

    # Output path
    print(
        f"Model artifacts will be saved to: "
        f"{sklearn_estimator.output_path}"
    )

except Exception as e:
    print(f"Error: {e}")