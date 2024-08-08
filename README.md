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
- densenetmodel.py contains a DenseNet-based transfer learning model for Stage 1 predictions (classification of each condition at each spinal level)
- Champion and Challenger models are in development for Stage 2 (prediction of severity for each condition at each spinal level)
- a Challenger model based on ResNet architecture is also in development for Stage 1 predicitons.

Model Training, Evaluation, and Predictions:
- trainer.py
- predict.py
- convert_predictions.ipynb

Environment, Variables, and Utilities:
- utils.py contains a class wrapper for a collection of helper functions used throughout the other code files
- constants.py defines a collection of variables used throughout the other code files
- environment.yml contains the setup for our virtual machine, including the specific versions of python packages used