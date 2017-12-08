---
title: Overview:
notebook: Report-Homepage.ipynb
nav_include: 1
---


## AC209A Data Science Final project
Jinwon Chung and Kate Zhou
Group #55


### Motivation:

Predicting and classifying human motion using wearable sensors have a wide variety of applications. These applications span from development of entertainment technology to assistive devices. One particularly interesting use case is in combining an gait classification model with soft exoskeletons in order to determine the best types of assistance at different walking or running conditions. In particular, the necessary assistance to the hip joint is highly dependent on the walking slope. In this project, we aim to develop a walking slope classifer.

Currently the hip-only soft exoskeleton suit from Harvard Biodesign Lab uses foot IMUs to estimate the slope of the ground; however, it would be preferred to remove this sensor in the suit to reduce suit complexity and donning time. There are other parameters the system collects, such as torso orientation and thigh angles, that could be used to predict walking slope. Our project aims to explore this possibility and build the best predictor possible using the data available.


### Problem Statement:
How to predict walking slope using Hip-Only exosuit data with no foot IMU?
1. solve as a classification problem of classifying flat ground, uphill or downhill. 
2. solve as a regression problem to estimate the exact walking slope and a classification problem with 7 classes at discrete slopes. Compare the performance of the two methods.

![IMUs](/Images/Jon.jpg)



