import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# AWS info
default_bucket = sagemaker_session.default_bucket()

account_id = sagemaker_session.boto_session.client(
    'sts'
).get_caller_identity()['Account']

# Model artifact location
MODEL_ARTIFACTS_S3 = (
    f"s3://{default_bucket}/models/local-trained/model.tar.gz"
)

# IAM role
SAGEMAKER_ROLE = (
    f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
)

# Unique endpoint name
ENDPOINT_NAME = "california-housing-serverless-endpoint"

try:
    # Create model object
    model = SKLearnModel(
        model_data=MODEL_ARTIFACTS_S3,
        role=SAGEMAKER_ROLE,
        entry_point='entry_point.py',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )

    # Serverless config
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=10
    )

    # Deploy model
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
        wait=False
    )

    # Get endpoint info
    endpoint_description = (
        sagemaker_session
        .sagemaker_client
        .describe_endpoint(
            EndpointName=ENDPOINT_NAME
        )
    )

    # Print status
    status = endpoint_description["EndpointStatus"]
    print(f"Endpoint status: {status}")

except Exception as e:
    print(f"Error: {e}")