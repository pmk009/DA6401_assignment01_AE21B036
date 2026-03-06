# Assignment 1: Multi-Layer Perceptron for Image Classification

**Name:** Kishore M  
**Roll Number:** AE21B036
**Course:** DA6401- Introduction to Deep Learning (Jan-May 2026)

---
## Weights & Biases Report

All experiments and hyperparameter sweeps are logged using **Weights & Biases**.

W&B Report:  
https://wandb.ai/DA6401_assignment01/assignment_01/reports/DA6401-Assignment01--VmlldzoxNjA0NjIxMg

---

## GitHub Repository

Project Repository:  
https://github.com/pmk009/DA6401_assignment01_AE21B036

---

## Overview

This project implements a **Multi-Layer Perceptron (MLP) from scratch using NumPy**.  
All core neural network components including layers, activation functions, loss functions, optimizers, and backpropagation are implemented manually without using deep learning frameworks.

The model is trained and evaluated on the **MNIST** and **Fashion-MNIST** datasets.

---

## Features

- Neural network implemented **entirely using NumPy**
- Manual implementation of **forward propagation and backpropagation**
- Multiple **activation functions**
- Multiple **optimization algorithms**
- Training and evaluation on **MNIST / Fashion-MNIST**
- Experiment tracking using **Weights & Biases (wandb)**

---

## Implemented Components

### Activation Functions
- Sigmoid
- Tanh
- ReLU
- Softmax

### Optimizers
- SGD
- Momentum
- Nesterov Accelerated Gradient (NAG)
- RMSProp
- Adam
- Nadam

### Loss Function
- Mean Squared Error
- Cross Entropy 

### Dataset
- MNIST
- Fashion-MNIST

---

## Installation

Clone the repository:

```bash
git clone https://https://github.com/pmk009/DA6401_assignment01_AE21B036.git
cd DA6401_assignment01_AE21B036
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Training Script

Example command:

```bash
python train.py \
--dataset mnist \
--epochs 10 \
--batch_size 32 \
--learning_rate 0.001 \
--optimizer adam \
--num_layers 2 \
--hidden_size 128 128\
--activation relu
```

---

## Learning Outcomes

Through this assignment, the following concepts were implemented and understood:

- Forward and backward propagation in neural networks
- Manual gradient computation
- Implementation of optimization algorithms
- Hyperparameter tuning
- Experiment tracking using Weights & Biases

---