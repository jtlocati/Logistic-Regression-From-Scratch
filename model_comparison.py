import numpy
from sklearn.linear_model import LogisticRegression
from model.logistic_regression import TrainLoop
from testing.SanityCheck import ds_forward_demo, ds_separable
import warnings
#suppress futre warning coming from sklearn model initalization
warnings.filterwarnings("ignore", category=FutureWarning)

#initalize genral datasets:
X,y = ds_separable()

#Train coustom model to reach both test types:\
w_coustom, b_coustom, losses_coustom = TrainLoop(X, y, learning_rate=0.1, epochs=3000, lambda_12=0.0)

#train sklearn model to reach both test types:
model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
model.fit(X, y)
w_sklearn = model.coef_.reshape(-1)
b_sklearn = model.intercept_[0]

#Choose test type
TestType = int(input("Simple Test: 1 \nNoise Test: 2\n>"))


if TestType == 1:


    print("========Coustom Model Implementation==================")
    print(f"Weights {w_coustom}\nBias {b_coustom}\nlosses {losses_coustom[-1]}")

    #Train sklearn implementation

    print("==================Sklearn Model Implementation==================")
    print(f"Weights: {w_sklearn}\nBias: {b_sklearn}\nlosses: {model.score(X,y)}")

    print("===================Dot Product Similarity Check==================")
    print(f"Scrach norm: {numpy.linalg.norm(w_coustom)}     ||||     sklearn norm: {numpy.linalg.norm(w_sklearn)}")

elif TestType == 2:
    #initalize larger DS with noise
    rng = numpy.random.default_rng(42)

    n = 500

    # Class 0 centered at (-1, -1)
    X0 = rng.normal(loc=-1.0, scale=1.0, size=(n//2, 2))

    # Class 1 centered at (1, 1)
    X1 = rng.normal(loc=1.0, scale=1.0, size=(n//2, 2))

    X = numpy.vstack([X0, X1])
    y = numpy.array([0]*(n//2) + [1]*(n//2), dtype=float)

    # Shuffle
    indices = rng.permutation(n)
    X = X[indices]
    y = y[indices]

    #standarizeing features for fair comparison:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    #comparison outputs:
    print("===================Comparison===================")

    print("Custom accuracy:", numpy.mean((1 / (1 + numpy.exp(-(X @ w_coustom + b_coustom))) >= 0.5) == y))
    print("Sklearn accuracy:", model.score(X, y))

    print("\nWeight norms:")
    print("Scratch:", numpy.linalg.norm(w_coustom))
    print("Sklearn:", numpy.linalg.norm(w_sklearn))

    # Cosine similarity between weight vectors
    cos_sim = numpy.dot(w_coustom, w_sklearn) / (
        numpy.linalg.norm(w_coustom) * numpy.linalg.norm(w_sklearn)
    )

    print("\nCosine similarity (direction match):", cos_sim)
