import papermill as pm
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"executed_notebooks/modelPrototype_1_output_{timestamp}.ipynb"

pm.execute_notebook(
    input_path="modelPrototype.ipynb",
    output_path=output_path,
    parameters={
        "bucket_name": "churn-model-prediction",
        "source_blob_name": "Telecom-Customer-Churn.csv",
        "destination_file_name": "/home/jupyter/ChurnPropensity/dataset1/Telecom-Customer-Churn.csv"
    },
    kernel_name="python3"  # Use this kernel name
)
