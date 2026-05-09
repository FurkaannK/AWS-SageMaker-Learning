import sagemaker
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute, OutputDataConfig

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get AWS configuration
default_bucket = sagemaker_session.default_bucket()
account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
region = sagemaker_session.boto_region_name

# Configuration constants
S3_TRAIN_DATA_URI = (
    f"s3://{default_bucket}/datasets/california_housing_train.csv"
)

SAGEMAKER_ROLE = (
    f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
)

MODEL_OUTPUT_PATH = (
    f"s3://{default_bucket}/models/california-housing/"
)

INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 30

# Retrieve sklearn training image
sklearn_image = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type=INSTANCE_TYPE
)

# Source code configuration
source_code = SourceCode(
    source_dir=".",
    entry_script="train.py"
)

# Compute configuration
compute_config = Compute(
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    volume_size_in_gb=VOLUME_SIZE_GB
)

# Output configuration
output_config = OutputDataConfig(
    s3_output_path=MODEL_OUTPUT_PATH
)

# Initialize ModelTrainer
model_trainer = ModelTrainer(
    training_image=sklearn_image,
    source_code=source_code,
    base_job_name="sklearn-modeltrainer",
    role=SAGEMAKER_ROLE,
    compute=compute_config,
    output_data_config=output_config
)

# Input data configuration
input_data = [
    InputData(
        channel_name="train",
        data_source=S3_TRAIN_DATA_URI
    )
]

try:
    # Start training job asynchronously
    model_trainer.train(
        input_data_config=input_data,
        wait=False
    )

    # Access latest training job
    latest_job = model_trainer._latest_training_job

    # Print training job information
    print(f"Training Job Name: {latest_job.training_job_name}")

    print(f"Training Job Status: {latest_job.training_job_status()}")

except Exception as e:
    print(f"Error: {e}")