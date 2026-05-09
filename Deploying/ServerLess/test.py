import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker.predictor import Predictor

# Endpoint name
ENDPOINT_NAME = "california-housing-serverless-endpoint"

# Test data path
TEST_DATA_FILE = "data/california_housing_test.csv"

try:
    # Connect to endpoint
    predictor = Predictor(endpoint_name=ENDPOINT_NAME)

    # Serializer & Deserializer
    predictor.serializer = CSVSerializer()
    predictor.deserializer = CSVDeserializer()

    # Load test data
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_test = test_df.drop("MedHouseVal", axis=1)
    y_test = test_df["MedHouseVal"]

    # Make predictions
    predictions = predictor.predict(X_test.values).flatten()

    # Evaluation
    test_r2 = r2_score(y_test, predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"R² Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")

except Exception as e:
    print(f"Error: {e}")