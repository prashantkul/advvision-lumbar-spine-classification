# advvision-lumbar-spine-classification

## Getting Started

This repo contains work done for the Computer Vision project at Univeristy of Chicago Master of Science.

## Dataset

The dataset used for this project is available at: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

## Image format

This data uses the DICOM format for the images. https://www.dicomstandard.org/

## Labels

There are overall 75 labels for this dataset. 25 main labels and 3 secondary labels.

### Following are the 25 primary labels for this dataset:
* spinal_canal_stenosis_l1_l2
* spinal_canal_stenosis_l2_l3
* spinal_canal_stenosis_l3_l4
* spinal_canal_stenosis_l4_l5
* spinal_canal_stenosis_l5_s1
* left_neural_foraminal_narrowing_l1_l2
* left_neural_foraminal_narrowing_l2_l3
* left_neural_foraminal_narrowing_l3_l4
* left_neural_foraminal_narrowing_l4_l5
* left_neural_foraminal_narrowing_l5_s1
* right_neural_foraminal_narrowing_l1_l2
* right_neural_foraminal_narrowing_l2_l3
* right_neural_foraminal_narrowing_l3_l4
* right_neural_foraminal_narrowing_l4_l5
* right_neural_foraminal_narrowing_l5_s1
* left_subarticular_stenosis_l1_l2
* left_subarticular_stenosis_l2_l3
* left_subarticular_stenosis_l3_l4
* left_subarticular_stenosis_l4_l5
* left_subarticular_stenosis_l5_s1
* right_subarticular_stenosis_l1_l2
* right_subarticular_stenosis_l2_l3
* right_subarticular_stenosis_l3_l4
* right_subarticular_stenosis_l4_l5
* right_subarticular_stenosis_l5_s1

* Each of this labels have 3 sub-labels: 
  * Normal/Mild 
  * Moderate 
  * Severe

# Format of Github Repo
Folder and File Descriptions
1. data/:
    * raw/: Contains the raw dataset files.
    * processed/: Contains processed datasets ready for analysis.
2. notebooks/:
    * 1_EDA.ipynb: Notebook for Exploratory Data Analysis.
    * 2_Model_Training.ipynb: Notebook for Model Training and Evaluation.
    * 3_Model_Operations.ipynb: Notebook for Model Operations and deployment plans.
3. reports/:
    * Final_Report.md: The final compiled report.
    * abstract.md: Abstract of the project.
    * eda.md: Detailed report on Exploratory Data Analysis.
    * model_training.md: Report on Model Training and Evaluation.
    * model_operations.md: Report on Model Operations and deployment plans.
    * conclusion.md: Conclusion of the project findings.
4. src/:
    * data_preprocessing.py: Scripts for preprocessing the data.
    * model_training.py: Scripts for training the models.
    * model_evaluation.py: Scripts for evaluating the models.
    * model_operations.py: Scripts for deployment and maintenance of the models.
5. requirements.txt: List of dependencies for the project, installable via pip.
6. environment.yml: Conda environment configuration file.
7. README.md: Overview of the project, setup instructions, and any other relevant information.
8. .gitignore: Specifies files and directories to be ignored by git (e.g., data files, environment files).
9. LICENSE: License file for the project.

