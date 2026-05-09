import sagemaker
from sagemaker.model import ModelPackage
from sagemaker.serverless import ServerlessInferenceConfig

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# Get AWS account ID
account_id = sagemaker_session.account_id()

# SageMaker execution role
SAGEMAKER_ROLE = (
    f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
)

# Model package group
MODEL_PACKAGE_GROUP_NAME = (
    "california-housing-pipeline-models"
)

# Endpoint name
ENDPOINT_NAME = "california-housing-estimator"

try:

    # Get SageMaker client
    sagemaker_client = sagemaker_session.sagemaker_client

    # List approved model packages
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending'
    )

    # Extract model package list
    model_packages = response.get(
        'ModelPackageSummaryList',
        []
    )

    # Latest approved model package ARN
    model_package_arn = (
        model_packages[0]['ModelPackageArn']
    )

    # Create ModelPackage object
    model = ModelPackage(
        role=SAGEMAKER_ROLE,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session
    )

    # Configure serverless inference
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=10
    )

    # Deploy endpoint
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
        wait=False
    )

    # Describe endpoint
    endpoint_description = (
        sagemaker_session
        .sagemaker_client
        .describe_endpoint(
            EndpointName=ENDPOINT_NAME
        )
    )

    # Endpoint status
    status = endpoint_description['EndpointStatus']

    print(f"Endpoint status: {status}")

except Exception as e:
    print(f"Error: {e}")