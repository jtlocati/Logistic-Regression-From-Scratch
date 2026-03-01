# Logistic Regression From Scratch (NumPy)

## Overview

This project implements **binary logistic regression entirely from scratch using NumPy** — including forward propagation, binary cross-entropy loss, gradient descent optimization, L2 regularization, evaluation metrics, and comparison against `scikit-learn`.

The goal of this project was not just to *use* logistic regression, but to fully understand:

- How probabilities are modeled with the sigmoid function  
- How cross-entropy measures prediction error  
- How gradients are derived and used in optimization  
- How regularization affects learned parameters  
- How a custom implementation compares to a production ML library  

---

## What Logistic Regression Is

Logistic regression is a binary classification algorithm used to predict the probability that an input belongs to class 1 (vs class 0).

Unlike linear regression, logistic regression outputs a **probability between 0 and 1**, using the sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

If the predicted probability is ≥ 0.5, we classify as 1. Otherwise, 0.

---

## Model Formulation

### Linear Model

\[
z = Xw + b
\]

- `X` → input feature matrix (m samples × n features)  
- `w` → weight vector (n,)  
- `b` → bias scalar  

### Probability Output

\[
\hat{y} = \sigma(z)
\]

Each prediction represents:

\[
P(y = 1 \mid x)
\]

---

## Loss Function (Binary Cross-Entropy)

Binary cross-entropy measures how wrong the predicted probabilities are:

\[
L = -\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
\right]
\]

Why this works:

- It strongly penalizes confident incorrect predictions  
- It produces clean, well-behaved gradients  
- It is the standard loss for probabilistic binary classification  

---

## Gradient Derivation

The gradients used for optimization are:

\[
dz = \hat{y} - y
\]

\[
dw = \frac{1}{m} X^T dz
\]

\[
db = \frac{1}{m} \sum dz
\]

Interpretation:

- `dz` → error signal (prediction − truth)  
- `dw` → distributes error back through features  
- `db` → shifts the decision boundary  

These gradients are used in **gradient descent**:

\[
w := w - \alpha dw
\]

\[
b := b - \alpha db
\]

Where `α` is the learning rate.

---

## L2 Regularization

To prevent overly large weights and improve generalization:

\[
L_{total} = L + \frac{\lambda}{2m} \sum w^2
\]

This adds the following to the weight gradient:

\[
\frac{\lambda}{m} w
\]

Regularization reduces weight magnitude while preserving the direction of the decision boundary.

---

## Project Features

- Fully vectorized NumPy implementation  
- Numerically stable sigmoid  
- Gradient descent training loop  
- Optional L2 regularization  
- Accuracy / Precision / Recall metrics  
- Loss curve plotting  
- Comparison against sklearn implementation  
- Synthetic noisy dataset evaluation  

---

## Comparison vs scikit-learn

Two experiments were performed to validate correctness.

### 1️⃣ Simple Separable Dataset

**Custom Implementation**


Weights: [2.862, 2.862]
Loss: 0.00163


**sklearn Implementation**


Weights: [4.267, 4.267]
Accuracy: 1.0


Both models achieve perfect accuracy.  
Magnitude differences are expected due to solver differences and optimization strategies.

---

### 2️⃣ Noisy Dataset (More Realistic Scenario)


Custom accuracy: 0.928
Sklearn accuracy: 0.928
Cosine similarity: 0.9999999999999999


Key takeaway:

- Accuracy matches exactly  
- Weight vectors align almost perfectly (cosine similarity ≈ 1.0)  
- Norm differences are expected due to solver and regularization conventions  

This confirms the correctness of the gradient implementation.

---

## How To Run

Create a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

Run sanity checks:

python sanity_check.py

Run sklearn comparison:

python model_comparison.py