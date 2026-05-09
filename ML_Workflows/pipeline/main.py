import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Create SageMaker sessions
sagemaker_session = sagemaker.Session()
pipeline_session = PipelineSession()

# AWS account & bucket
account_id = sagemaker_session.account_id()
default_bucket = sagemaker_session.default_bucket()

# IAM role
SAGEMAKER_ROLE = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"

# Pipeline name
PIPELINE_NAME = "california-housing-evaluation-pipeline"

# ---------------------------
# STEP 1: DATA PROCESSING
# ---------------------------
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="ProcessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{default_bucket}/datasets/california_housing.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train"
        ),
        ProcessingOutput(
            output_name="test_data",
            source="/opt/ml/processing/test"
        )
    ],
    code="data_processing.py"
)

# ---------------------------
# STEP 2: TRAINING
# ---------------------------
estimator = SKLearn(
    entry_point="train.py",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=pipeline_session
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
        )
    }
)

# ---------------------------
# STEP 3: EVALUATION
# ---------------------------
evaluation_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session
)

evaluation_report_property = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    code="evaluation.py",
    property_files=[evaluation_report_property]
)

# ---------------------------
# PIPELINE DEFINITION
# ---------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[processing_step, training_step, evaluation_step],
    sagemaker_session=sagemaker_session
)

# ---------------------------
# EXECUTION
# ---------------------------
try:
    pipeline.upsert(role_arn=SAGEMAKER_ROLE)

    execution = pipeline.start()

    print(f"Pipeline execution ARN: {execution.arn}")

    execution_details = execution.describe()

    print(f"Status: {execution_details['PipelineExecutionStatus']}")

except Exception as e:
    print(f"Error: {e}")