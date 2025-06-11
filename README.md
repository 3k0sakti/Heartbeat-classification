A deep learning approach for ECG arrhythmia classification that combines handcrafted morphology features with Convolutional Neural Networks (CNN) for improved accuracy and interpretability.

This code implements a hybrid machine learning model for automated ECG arrhythmia detection and classification. The system processes ECG signals from the MIT-BIH Arrhythmia Database and classifies heartbeats into 5 AAMI (Association for the Advancement of Medical Instrumentation) standard classes.

**Feature Categories (37 Total)**

Statistical Features (8): Mean, standard deviation, variance, skewness, kurtosis, min/max values
Peak Detection (4): Number of peaks, R-peak amplitude and position, QRS duration
QRS Complex (2): QRS width and area
P & T Waves (6): Amplitude and area measurements
Slope Features (4): Gradient-based characteristics
Interval Features (2): Pre and post R-peak intervals
Energy Features (3): Signal energy and RMS values
Clinical Features (8): Ratios, symmetry, and temporal characteristics
