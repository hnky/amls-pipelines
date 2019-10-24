import azureml.core
from azureml.core import Workspace, Datastore, Experiment
from azureml.core.model import Model
from azureml.core.compute import ComputeTarget, AmlCompute, DataFactoryCompute
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.data.data_reference import DataReference
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import DataTransferStep, PythonScriptStep, EstimatorStep
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.train.dnn import TensorFlow

clusterName = "NV6AICluster"

# Load workspace
ws = Workspace.from_config()

# Connect to Compute Target
computeCluster = ComputeTarget(workspace=ws, name=clusterName)

# connect to datastores
source_ds = Datastore.get(ws, 'SimpsonDataStore')
training_ds = Datastore.get(ws, 'SimpsonTrainingDataStore')

source_dataset = DataPath(datastore=source_ds, path_on_datastore="trainingdata")

# Parameters make it easy for us to re-run this training pipeline, including for retraining.
source_dataset_param = (PipelineParameter(name="source_dataset",default_value=source_dataset),
                          DataPathComputeBinding())

script_folder = "./steps"

# == Step 1 ==
cd = CondaDependencies.create(pip_packages=["azureml-sdk","opencv-python"])
amlcompute_run_config = RunConfiguration(conda_dependencies=cd)

training_data_location = PipelineData(name="trainingdata", datastore=training_ds)

preProcessDataStep = PythonScriptStep(name="Pre-process data",
                            script_name="prep.py",
                            compute_target=computeCluster,
                            runconfig=amlcompute_run_config,
                            inputs=[source_dataset_param],
                            arguments=['--source_path', source_dataset_param,
                                       '--destination_path', training_data_location,
                                        "--pic_size",64
                                      ],
                            outputs=[training_data_location],
                            source_directory=script_folder)



# == Step 2 ==
model = PipelineData(name="model", datastore=training_ds, output_path_on_compute="model")

script_params = [
    "--data-folder", training_data_location,
    "--output-folder", model,
    "--pic-size",64,
    "--batch-size",32,
    "--epochs",15
]

trainEstimator = TensorFlow(
                 source_directory = script_folder,
                 compute_target = computeCluster,
                 entry_script = "train.py", 
                 use_gpu = True,
                 use_docker = True,
                 conda_packages=["keras==2.2.2","opencv==3.4.2","scikit-learn"],
                 framework_version="1.10"
            )

trainOnGpuStep = EstimatorStep(
    name='Train Estimator Step',
    estimator=trainEstimator,
    inputs=[training_data_location],
    outputs=[model],
    compute_target=computeCluster,
    estimator_entry_script_arguments = script_params
)

# == Step 3 ==
model_name = "MargeOrHomer"
model_id = PipelineData(name="modelId", datastore=training_ds)

registerStep = PythonScriptStep(name="Register model for deployment",
                            script_name="register.py",
                            compute_target=computeCluster,
                            inputs=[model],
                            arguments=['--dataset_name', model_name,
                                       '--model_assets_path', model
                                      ],
                            outputs=[model_id],
                            source_directory=script_folder)

# Create the pipeline
prep_train_register = [preProcessDataStep,trainOnGpuStep,registerStep]
pipeline = Pipeline(workspace=ws, steps=[prep_train_register])
pipeline.validate()

# Publish the pipeline
mlpipeline = pipeline.publish(name="Marge Or Homer - Training pipeline")
print("Pipeline Published ID:"+mlpipeline.id)

# Submit the pipeline to be run
mlpipeline.submit(ws,"Marge-or-Homer" ,pipeline_parameters={"source_dataset":DataPath(datastore=source_ds, path_on_datastore="trainingdata")})