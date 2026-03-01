## Logistic Regression
Logistic regression is a method used in statistics in order to predict the outcome of somthing given the input of one or more things. A logistic regression function takes an value of type int and outputs a boolean (0 or 1), from this we fit a logistic function between the inputs and outputs wich allows us to make predictions of the possibility of an event happening. This logistic curve is then assigned a p value wich specifys how sensitve the model is to displaying true of false (simmalar to a T value in tokenization).

# Logistic Regression From Scratch (NumPy)

## What this is
A binary classifier implemented using only NumPy for training:
- sigmoid
- binary cross-entropy loss
- gradient descent
- (later) L2 regularization + evaluation metrics
- (later) comparison vs sklearn

## Why do it manually?
This allows my to learn and understand the math behind:
- probabilities (sigmoid)
- optimization (gradient descent)
- loss functions (cross-entropy)

## Core Math

Sigmoid: Translates real-valued input into a binary probibility
σ(z) = 1 / (1 + e^-z)

Model:
z = Xw + b
ŷ = σ(z)

Loss (Binary Cross-Entropy): Measures the amount of outcomes are incorectly predicted. 
L = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

Gradients:
dz = ŷ - y  -> Gives error signal by how far the predicted value is from the real label in the test set
dw = (1/m) X^T dz   -> Distibutes error back through the feature to show how the weights should change
db = (1/m) Σ dz    -> Averages the error across all samples to represent how the bias should shift. 