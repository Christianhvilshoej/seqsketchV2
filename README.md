# CS492D Course Project: Sequential Sketch Stroke Generation

Course project for CS492(D): Diffusion Models and Their Applications

[Project Topic](https://github.com/KAIST-Visual-AI-Group/Diffusion-Project-Drawing)

## Get Started

1. **Set Up Environment and Install Requirements**

   - Create the Conda environment:
     ```bash
     make create_conda_environment
     ```
   - Install the required dependencies:
     ```bash
     make install_requirements
     ```
   - `seqsketch` is added as an editable package during installation, making it easy to modify and develop.

2. **Download Data**

   - Download the raw data using the following command:
     ```bash
     make download_quickdraw
     ```
     It adds .ndjson files to the data folders.

3. **Train and Inference**

   - To start training, use commands like:
     ```bash
     train --config baseline.yaml
     ```
   - For inference, specify the model folder created during training. For example:
     ```bash
     inference --model_folder models/baseline/v0_20241101_195027
     ```
