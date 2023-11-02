# bart-hf-model
Bart Hugging-Face model for Eng to LTL translation.

You can refer to the following scripts for:

- Training model: train-model.py
- Inference of a Dataset: inference.py (it just skips the training in train-model.py and uses preloaded checkpoint from)
- Search for hyperparameters: hyperparameter-search.py (you will need wandb api key to use the sweep api)
- Generate test datasets: generate-data.py

### Installation/Setup
I have attached the requirements.txt and conda_dependencies_list.txt, but all you really need to do is the following:

```
conda install -c conda-forge spot
pip install transformers==4.33.1 datasets==2.14.5 evaluate==0.4.0 sacrebleu==2.3.1 accelerate==0.22.0 wandb==0.15.12 python-dotenv os numpy csv datetime
```
I ran above in the continuumio/anaconda3 docker image and it worked fine.

You can also refer to the singularity recipe I used if you want to use a pytorch-2.1.0-runtime singularity container:

```
Bootstrap: localimage
From: /usr/local/containers/pytorch-2.1.0-runtime.sif

%environment
        export LC_ALL=C

%post
        # Non interactive installs
        export DEBIAN_FRONTEND=noninteractive

        # Resynchronize the package index
        apt-get -y update

        # Add more packages after here
        
        export PATH=/opt/conda/bin:$PATH

        conda install -y -c conda-forge spot

        pip install transformers==4.33.1 datasets==2.14.5 evaluate==0.4.0 sacrebleu==2.3.1 accelerate==0.22.0 wandb==0.15.12 python-dotenv numpy
```

You may need to set paths for the code to work the way you need. Please review the scripts and change the variables as you see fit.

Also note I have only tested this on linux, so I'm not sure if there will be any issues with MAC or Windows.