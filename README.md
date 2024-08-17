# Lumbar Spine Degenerative Disease Classification with Deep Neural Networks
Prashant Kulkarni, Kwaku Ofori-Atta, Dharti Seagraves, Steve Veldman

## Summary

This repository contains code and other resources for our final project for ADSP 31023 - Advanced Computer Vision with Deep Learning, part of the Univeristy of Chicago's Master of Science in Applied Data Science program. This will also serve as the starting point for our submission to the associated Kaggle competition for this dataset, which will serve as a continuation of our group's work after completion of ADSP 31023.

## Dataset
The poject utilizes the RSNA 2024 Lumbar Spine Degenerative Classification dataset, available at:
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

## Additional Details
A full summary of the dataset, problem statement, and our work for the project can be found in our presentation deck:
https://docs.google.com/presentation/d/1yqncDNBqYR_Ylbd8w7Cb3hsD7GgMLR9w/edit?usp=sharing&ouid=117738997373544471509&rtpof=true&sd=true

## Guide to this Repository (Essential Code):
Data Exploration and Preprocessing:
- eda.ipynb contains our exploratory data analysis
- imageloader.py defines a class for loading our dataset, establishing the train/validate/test subsets, and application of Gaussian Attention Mask to each image

Models:
- Transfer learning models for Stage 1 predictions (classification of each condition at each spinal level) can be found in the vision_models folder
- Additional models (Stage 0, Stage 2, auxilary models) are located in subfolders within vision_models

Model Training, Evaluation, and Predictions:
- trainer.py and resnettrainer.py contain class wrappers for methods used in trainign the two Stage 1 transfer learning models
- predict.py contains class wrappers for methods used to generate predictions from and evaluate trained models

Environment, Variables, and Utilities:
- environment.yml contains the setup for our virtual machine, including the specific versions of python packages used