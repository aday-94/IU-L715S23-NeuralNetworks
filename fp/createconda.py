""" IUB - L715NNs/Nahuatl Spring 2023 - Final Project - Andrew Davis - How to create a MiniConda Environment:

In Google Colab (Per Chat GPT & it works):

Sure, here are the steps to install and use Miniconda3 in Google Colab:

    First, you need to download the Miniconda installer script for Linux from the official website. You can download it using the following command:

    arduino

!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Next, you need to run the installer script by running the following command:

bash

!bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local

This will install Miniconda in the /usr/local directory of the Colab environment.

Once Miniconda is installed, you can add it to your system path by running the following command:

go

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

This will allow you to use the conda command in your Colab notebook.

Finally, you can create a new Conda environment by running the following command:

lua

!conda create --name env_name

Replace env_name with the name you want to give to your environment.

Once the environment is created, you can activate it by running the following command:

diff

    !conda activate env_name

    Now you are inside the environment and can install the required packages using Conda or pip.

Note: Make sure to restart the runtime of your Colab notebook after running the above commands to apply the changes.

**************

Conda environment Via Rob P

Carbonate Option:

!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

then once you o through the prompts to install conda, create a new env with:
conda create -n <environment name> python=3.8


then:
conda activate <environment name> (andy_neural.venv is my env name)

^if that doesn't work , check path 

cat ~/.bash_profile 
 
Solve with :
source ~/.bashrc

And check with:
conda


I now have a Conda node that I can use for Nahuatl on the GPU, nn.final.venv and It has opennmt in it 

(base) [zleech@h2 miniconda3]$ conda activate nn.final.venv
(nn.final.venv) [zleech@h2 miniconda3]$ pip install OpenNMT-py (activate (no longer need to install))

"""