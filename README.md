An Exemplary Study of Machine Learning Failures Outside Training Fields
This repository contains the code and generated images for my study on neural network extrapolation limits. I trained a deep neural network to predict Reference Crop Evaporation (ERC) using the physical FAO56 formula. I then tested the model on data outside its training fields to observe its failures.

Project Overview
Machine learning models are highly capable of approximating complex mathematical functions. However, they lack an inherent understanding of physical laws. This project demonstrates what happens when a neural network is forced to make predictions outside the data boundaries it saw during its training phase.

The study includes:

Generating a synthetic dataset of 500,000 samples using Latin Hypercube Sampling.

Calculating the target ERC values using the standard physical equation.

Training a deep neural network to predict ERC based on 10 meteorological features.

Evaluating the model on In Distribution data.

Stress testing the model on three Out of Distribution scenarios: Extreme Arid, High Altitude, and Tropical Monsoon.

Generating comprehensive visualizations and a dynamic animation to illustrate the error limits.

Requirements
To run the code, you need Python installed along with the following libraries:

numpy

pandas

tensorflow

scipy

scikit learn

matplotlib

Usage
Clone this repository to your local machine.

Ensure you have the required libraries installed.

Run the main Python script to generate the dataset, train the model, run the tests, and save all figures.

Bash
python erc_baseline_fao56_fully_automated_fixed_axes.py
The script will automatically create a directory to save the dataset, the trained model, the scaler object, all 8 analysis figures, and the 10 dimensional error journey animation video.

Repository Contents
The main Python script containing the physics engine, model architecture, and plotting functions.

Figure 1: Training History

Figure 2: In Distribution Test Results

Figure 3: Input Feature Ranges

Figure 4: True vs Predicted for Scenarios Outside Training Fields

Figure 5: Residual Distributions Outside Training Fields

Figure 6: MAE Comparison

Figure 7: Absolute Error vs Key Features

Figure 8: Domain Limitations

Video 1: Model Stress Test Animation

You can access all the coding and generated images from this GitHub address. If you see an error, please report it.
