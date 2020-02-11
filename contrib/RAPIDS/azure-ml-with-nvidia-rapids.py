# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.
# %% [markdown]
# ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/contrib/RAPIDS/azure-ml-with-nvidia-rapids/azure-ml-with-nvidia-rapids.png)
# %% [markdown]
# # NVIDIA RAPIDS in Azure Machine Learning
# %% [markdown]
# The [RAPIDS](https://www.developer.nvidia.com/rapids) suite of software libraries from NVIDIA enables the execution of end-to-end data science and analytics pipelines entirely on GPUs. In many machine learning projects, a significant portion of the model training time is spent in setting up the data; this stage of the process is known as Extraction, Transformation and Loading, or ETL. By using the DataFrame API for ETLÂ and GPU-capable ML algorithms in RAPIDS, data preparation and training models can be done in GPU-accelerated end-to-end pipelines without incurring serialization costs between the pipeline stages. This notebook demonstrates how to use NVIDIA RAPIDS to prepare data and train modelÃ‚Â in Azure.
#  
# In this notebook, we will do the following:
#  
# * Create an Azure Machine Learning Workspace
# * Create an AMLCompute target
# * Use a script to process our data and train a model
# * Obtain the data required to run this sample
# * Create an AML run configuration to launch a machine learning job
# * Run the script to prepare data for training and train the model
#  
# Prerequisites:
# * An Azure subscription to create a Machine Learning Workspace
# * Familiarity with the Azure ML SDK (refer to [notebook samples](https://github.com/Azure/MachineLearningNotebooks))
# * A Jupyter notebook environment with Azure Machine Learning SDK installed. Refer to instructions to [setup the environment](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local)
# %% [markdown]
# ### Verify if Azure ML SDK is installed

# %%
import azureml.core
print("SDK version:", azureml.core.VERSION)


# %%
import os
from azureml.core import Workspace, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.data.data_reference import DataReference
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.widgets import RunDetails

# %% [markdown]
# ### Create Azure ML Workspace
# %% [markdown]
# The following step is optional if you already have a workspace. If you want to use an existing workspace, then
# skip this workspace creation step and move on to the next step to load the workspace.
#  
# <font color='red'>Important</font>: in the code cell below, be sure to set the correct values for the subscription_id, 
# resource_group, workspace_name, region before executing this code cell.

# %%
# subscription_id = os.environ.get("SUBSCRIPTION_ID", "73612009-b37b-413f-a3f7-ec02f12498cf")
# resource_group = os.environ.get("RESOURCE_GROUP", "RAPIDS-MD")
# workspace_name = os.environ.get("WORKSPACE_NAME", "RAPIDS-DemoretM")
# workspace_region = os.environ.get("WORKSPACE_REGION", "West US 2")

# ws = Workspace.create(workspace_name, subscription_id=subscription_id, resource_group=resource_group, location=workspace_region)

# # write config to a local directory for future use
# ws.write_config()

# %% [markdown]
# ### Load existing Workspace

# %%
ws = Workspace.from_config()

# if a locally-saved configuration file for the workspace is not available, use the following to load workspace
# ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

scripts_folder = "scripts_folder"

if not os.path.isdir(scripts_folder):
    os.mkdir(scripts_folder)

# %% [markdown]
# ### Create AML Compute Target
# %% [markdown]
# Because NVIDIA RAPIDS requires P40 or V100 GPUs, the user needs to specify compute targets from one of [NC_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#ncv3-series), [NC_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#ncv2-series), [ND](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#nd-series) or [ND_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#ndv2-series-preview) virtual machine types in Azure; these are the families of virtual machines in Azure that are provisioned with these GPUs.
#  
# Pick one of the supported VM SKUs based on the number of GPUs you want to use for ETL and training in RAPIDS.
#  
# The script in this notebook is implemented for single-machine scenarios. An example supporting multiple nodes will be published later.

# %%
gpu_cluster_name = "gpucluster"

if gpu_cluster_name in ws.compute_targets:
    gpu_cluster = ws.compute_targets[gpu_cluster_name]
    if gpu_cluster and type(gpu_cluster) is AmlCompute:
        print('Found compute target. Will use {0} '.format(gpu_cluster_name))
else:
    print("creating new cluster")
    # vm_size parameter below could be modified to one of the RAPIDS-supported VM types
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "Standard_NC6s_v3", min_nodes=1, max_nodes = 1)

    # create the cluster
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, provisioning_config)
    gpu_cluster.wait_for_completion(show_output=True)

# %% [markdown]
# ### Script to process data and train model
# %% [markdown]
# The _process&#95;data.py_ script used in the step below is a slightly modified implementation of [RAPIDS Mortgage E2E example](https://github.com/rapidsai/notebooks-contrib/blob/master/intermediate_notebooks/E2E/mortgage/mortgage_e2e.ipynb).

# %%
# copy process_data.py into the script folder
import shutil
shutil.copy('./process_data.py', os.path.join(scripts_folder, 'process_data.py'))

# %% [markdown]
# ### Data required to run this sample
# %% [markdown]
# This sample uses [Fannie Mae's Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html). Once you obtain access to the data, you will need to make this data available in an [Azure Machine Learning Datastore](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data), for use in this sample. The following code shows how to do that.
# %% [markdown]
# ### Downloading Data

# %%
import tarfile
import hashlib
from urllib.request import urlretrieve

def validate_downloaded_data(path):
    if(os.path.isdir(path) and os.path.exists(path + '//names.csv')) :
        if(os.path.isdir(path + '//acq' ) and len(os.listdir(path + '//acq')) == 8):
            if(os.path.isdir(path + '//perf' ) and len(os.listdir(path + '//perf')) == 11):
                print("Data has been downloaded and decompressed at: {0}".format(path))
                return True
    print("Data has not been downloaded and decompressed")
    return False

def show_progress(count, block_size, total_size):
    global pbar
    global processed
    
    if count == 0:
        pbar = ProgressBar(maxval=total_size)
        processed = 0
    
    processed += block_size
    processed = min(processed,total_size)
    pbar.update(processed)

        
def download_file(fileroot):
    filename = fileroot + '.tgz'
    if(not os.path.exists(filename) or hashlib.md5(open(filename, 'rb').read()).hexdigest() != '82dd47135053303e9526c2d5c43befd5' ):
        url_format = 'http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/{0}.tgz'
        url = url_format.format(fileroot)
        print("...Downloading file :{0}".format(filename))
        urlretrieve(url, filename)
        # pbar.finish()
        print("...File :{0} finished downloading".format(filename))
    else:
        print("...File :{0} has been downloaded already".format(filename))
    return filename

def decompress_file(filename,path):
    tar = tarfile.open(filename)
    print("...Getting information from {0} about files to decompress".format(filename))
    members = tar.getmembers()
    numFiles = len(members)
    so_far = 0
    for member_info in members:
        tar.extract(member_info,path=path)
        so_far += 1
    print("...All {0} files have been decompressed".format(numFiles))
    tar.close()


# %%
fileroot = 'mortgage_2000-2001'
path = './{0}'.format(fileroot)
pbar = None
processed = 0

if(not validate_downloaded_data(path)):
    print("Downloading and Decompressing Input Data")
    filename = download_file(fileroot)
    decompress_file(filename,path)
    print("Input Data has been Downloaded and Decompressed")

# %% [markdown]
# ### Uploading Data to Workspace

# %%
ds = ws.get_default_datastore()

# download and uncompress data in a local directory before uploading to data store
# directory specified in src_dir parameter below should have the acq, perf directories with data and names.csv file

# ---->>>> UNCOMMENT THE BELOW LINE TO UPLOAD YOUR DATA IF NOT DONE SO ALREADY <<<<----
# ds.upload(src_dir=path, target_path=fileroot, overwrite=False, show_progress=True)

# data already uploaded to the datastore
data_ref = DataReference(data_reference_name='data', datastore=ds, path_on_datastore=fileroot)

# %% [markdown]
# ### Create AML run configuration to launch a machine learning job
# %% [markdown]
# RunConfiguration is used to submit jobs to Azure Machine Learning service. When creating RunConfiguration for a job, users can either 
# 1. specify a Docker image with prebuilt conda environment and use it without any modifications to run the job, or 
# 2. specify a Docker image as the base image and conda or pip packages as dependnecies to let AML build a new Docker image with a conda environment containing specified dependencies to use in the job
# 
# The second option is the recommended option in AML. 
# The following steps have code for both options. You can pick the one that is more appropriate for your requirements. 
# %% [markdown]
# #### Specify prebuilt conda environment
# %% [markdown]
# The following code shows how to install RAPIDS using conda. The `rapids.yml` file contains the list of packages necessary to run this tutorial. **NOTE:** Initial build of the image might take up to 20 minutes as the service needs to build and cache the new image; once the image is built the subequent runs use the cached image and the overhead is minimal.

# %%
cd = CondaDependencies(conda_dependencies_file_path='rapids.yml')
run_config = RunConfiguration(conda_dependencies=cd)
run_config.framework = 'python'
run_config.target = gpu_cluster_name
run_config.environment.docker.enabled = True
run_config.environment.python.user_managed_dependencies
# run_config.environment.docker.gpu_support = True
run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04"
run_config.environment.spark.precache_packages = False
run_config.data_references={'data':data_ref.to_config()}
test_name="rapidstest"

# %% [markdown]
# #### Using Docker
# %% [markdown]
# Alternatively, you can specify RAPIDS Docker image.

# %%
run_config = RunConfiguration()
run_config.framework = 'python'
run_config.environment.python.user_managed_dependencies = True
run_config.environment.python.interpreter_path = '/conda/envs/rapids/bin/python'
run_config.target = gpu_cluster_name
run_config.environment.docker.enabled = True
run_config.environment.docker.gpu_support = True
run_config.environment.docker.base_image = "rapidsai/rapidsai:cuda9.2-runtime-ubuntu18.04"
# run_config.environment.docker.base_image_registry.address = '<registry_url>' # not required if the base_image is in Docker hub
# run_config.environment.docker.base_image_registry.username = '<user_name>' # needed only for private images
# run_config.environment.docker.base_image_registry.password = '<password>' # needed only for private images
run_config.environment.spark.precache_packages = False
run_config.data_references={'data':data_ref.to_config()}
test_name="rapidstest_noconda"

# %% [markdown]
# ### Wrapper function to submit Azure Machine Learning experiment

# %%
# parameter cpu_predictor indicates if training should be done on CPU. If set to true, GPUs are used *only* for ETL and *not* for training
# parameter num_gpu indicates number of GPUs to use among the GPUs available in the VM for ETL and if cpu_predictor is false, for training as well 
def run_rapids_experiment(cpu_training, gpu_count, part_count):
    # any value between 1-4 is allowed here depending the type of VMs available in gpu_cluster
    if gpu_count not in [1, 2, 3, 4]:
        raise Exception('Value specified for the number of GPUs to use {0} is invalid'.format(gpu_count))

    # following data partition mapping is empirical (specific to GPUs used and current data partitioning scheme) and may need to be tweaked
    max_gpu_count_data_partition_mapping = {1: 3, 2: 4, 3: 6, 4: 8}
    
    if part_count > max_gpu_count_data_partition_mapping[gpu_count]:
        print("Too many partitions for the number of GPUs, exceeding memory threshold")
        
    if part_count > 11:
        print("Warning: Maximum number of partitions available is 11")
        part_count = 11
        
    end_year = 2000
    
    if part_count > 4:
        end_year = 2001 # use more data with more GPUs

    src = ScriptRunConfig(source_directory=scripts_folder, 
                          script='process_data.py', 
                          arguments = ['--num_gpu', gpu_count, '--data_dir', str(data_ref),
                                      '--part_count', part_count, '--end_year', end_year,
                                      '--cpu_predictor', cpu_training
                                      ],
                          run_config=run_config
                         )

    exp = Experiment(ws, test_name)
    run = exp.submit(config=src)
    RunDetails(run).show()
    return run

# %% [markdown]
# ### Submit experiment (ETL & training on GPU)

# %%
cpu_predictor = False
# the value for num_gpu should be less than or equal to the number of GPUs available in the VM
num_gpu = 1
data_part_count = 1
# train using CPU, use GPU for both ETL and training
run = run_rapids_experiment(cpu_predictor, num_gpu, data_part_count)

# %% [markdown]
# ### Submit experiment (ETL on GPU, training on CPU)
# 
# To observe performance difference between GPU-accelerated RAPIDS based training with CPU-only training, set 'cpu_predictor' predictor to 'True' and rerun the experiment

# %%
cpu_predictor = True
# the value for num_gpu should be less than or equal to the number of GPUs available in the VM
num_gpu = 1
data_part_count = 1
# train using CPU, use GPU for ETL
run = run_rapids_experiment(cpu_predictor, num_gpu, data_part_count)

# %% [markdown]
# ### Delete cluster

# %%
# delete the cluster
# gpu_cluster.delete()


