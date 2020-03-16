# Airbus Ship Detection

EE 435: Deep Learning Foundations from Scratch \
Northwestern Univertiy Winter 2020

## Memebers
* Blaine Rothrock
* Ilan Ponsky
* Will Dong

## Overview

Our group is interested in applying knowledge from this course to training TensorFlow models and getting a better understanding of deep artificial neural networks involved in image processing. In order to learn about this process in an organized and efficient way, we utilized a closed Kaggle competition that centered around our topic of image processing with neural networks. The competition we used was the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection). The goal of the Kaggle competition was for participants to be able to build a model that could “detect all ships in satellite images as quickly as possible.”

The specific goals we wanted to hit for our Deep Learning from Scratch final project were to: 
* Build a binary classifier model to gain a basic understanding of Tensorflow and how to build models in TensorFlow.
    - The objective of the binary classifier model is to output whether an image contained a ship or not utilizing optimization techniques in TensorFlow.
* Explore and implement a U-net model for image segmentation. 
    - U-Net models are the current state-of-art for image segmentation and where most started for this competition.
    - This is a hefty goal given the data size of ~40GB of images and the time it takes to training this complex model. Our goal is to build a model, attempt at training, and gain a understanding of the U-Net architecture.
    - To accomplish this, we utilized the notebook of Kevin Mader on Kaggle which served as an excellent foundation to get started with implementing the mentioned goals
