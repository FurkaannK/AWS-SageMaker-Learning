import sagemaker
from datetime import datetime, timedelta

PIPELINE_NAME = "california-housing-training-pipeline"

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Get the SageMaker client from the session
sagemaker_client = sagemaker_session.sagemaker_client

# List pipeline executions sorted by creation time (newest first)
response = sagemaker_client.list_pipeline_executions(
    PipelineName=PIPELINE_NAME,
    SortBy='CreationTime',
    SortOrder='Descending'
)

# Extract the execution summaries from the response
executions = response['PipelineExecutionSummaries']

# Check if any executions exist
if not executions:
    print("No pipeline executions found.")
else:
    # Get the latest execution (first item in the list)
    latest_execution = executions[0]
    
    # Extract execution details
    execution_arn = latest_execution['PipelineExecutionArn']
    status = latest_execution['PipelineExecutionStatus']
    start_time = latest_execution['StartTime']
    
    # Display basic execution information
    print(f"Latest Execution ARN: {execution_arn}")
    print(f"Status: {status}")
    print(f"Start Time: {start_time}")
    
    # Check if the execution has completed (EndTime exists)
    if 'EndTime' in latest_execution:
        end_time = latest_execution['EndTime']
        # Calculate how long the pipeline took to run
        duration = end_time - start_time
        print(f"End Time: {end_time}")
        print(f"Duration: {duration}")
    
    # Retrieve all steps for this specific execution
    steps_response = sagemaker_client.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn
    )
    
    # Extract the steps from the response
    steps = steps_response['PipelineExecutionSteps']
    
    # Iterate through each step and display its details
    for step in steps:
        # Extract step information
        step_name = step['StepName']
        step_status = step['StepStatus']
        step_start = step.get('StartTime', 'Not started')
        
        # Display step name and status
        print(f"\n{step_name}: {step_status}")
        
        # Show start time if the step has started
        if step_start != 'Not started':
            print(f"Started: {step_start}")
        
        # If step has completed, show end time and duration
        if 'EndTime' in step:
            step_end = step['EndTime']
            # Calculate step execution time
            step_duration = step_end - step['StartTime']
            print(f"Ended: {step_end}")
            print(f"Duration: {step_duration}")
        
        # For failed steps, show the failure reason to help with debugging
        if step_status == 'Failed' and 'FailureReason' in step:
            print(f"Failure Reason: {step['FailureReason']}")