import numpy
from sklearn.linear_model import LogisticRegression
from model.logistic_regression import TrainLoop
from testing.SanityCheck import ds_forward_demo, ds_separable
import warnings
#suppress futre warning coming from sklearn model initalization
warnings.filterwarnings("ignore", category=FutureWarning)

X,y = ds_separable()

# Train coustom implementation
w_coustom, b_coustom, losses_coustom = TrainLoop(X, y, learning_rate=0.1, epochs=3000, lambda_12=0.0)

print("========Coustom Model Implementation==================")
print(f"Weights {w_coustom}\nBias {b_coustom}\nlosses {losses_coustom[-1]}")

#Train sklearn implementation
model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
model.fit(X, y)

w_sklearn = model.coef_.reshape(-1)
b_sklearn = model.intercept_[0]

print("==================Sklearn Model Implementation==================")
print(f"Weights: {w_sklearn}\nBias: {b_sklearn}\nlosses: {model.score(X,y)}")

print("===================Dot Product Similarity Check==================")
print(f"Scrach norm: {numpy.linalg.norm(w_coustom)}     ||||     sklearn norm: {numpy.linalg.norm(w_sklearn)}")
