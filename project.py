import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ========================
# Load Data
# ========================
path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
df = pd.read_csv(os.path.join(path, "train.csv"))

# 🔥 Log transform target
y = np.log1p(df["SalePrice"])
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

# ========================
# Preprocessing
# ========================
numCols = X.select_dtypes(include=['number']).columns
catCols = X.select_dtypes(include=['object']).columns

# Fill missing values
X[numCols] = X[numCols].fillna(X[numCols].median())
X[catCols] = X[catCols].fillna("Missing")

# Optional ordinal mapping
qualMap = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "Missing": 0
}

ordinalCols = ["KitchenQual", "ExterQual"]
for col in ordinalCols:
    if col in X.columns:
        X[col] = X[col].map(qualMap)

# One-hot encoding
X = pd.get_dummies(X)

# ========================
# K-Fold Setup
# ========================
kFold = KFold(n_splits=5, shuffle=True, random_state=42)

ridgeRmseList = []
lassoRmseList = []
gbRmseList = []
hybridRmseList = []

# For residual plot
allYTest = []
allRidgePred = []
allHybridPred = []

# ========================
# RMSE Function (convert back)
# ========================
def rmse(yTrue, yPred):
    yTrueExp = np.expm1(yTrue)
    yPredExp = np.expm1(yPred)
    return np.sqrt(mean_squared_error(yTrueExp, yPredExp))

# ========================
# K-Fold Training
# ========================
for trainIdx, testIdx in kFold.split(X):

    XTrain, XTest = X.iloc[trainIdx], X.iloc[testIdx]
    yTrain, yTest = y.iloc[trainIdx], y.iloc[testIdx]

    # Scaling for linear models
    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    # ------------------------
    # Ridge
    # ------------------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(XTrainScaled, yTrain)
    yPredRidge = ridge.predict(XTestScaled)

    # ------------------------
    # Lasso (tuned)
    # ------------------------
    lasso = Lasso(alpha=0.0005, max_iter=10000)
    lasso.fit(XTrainScaled, yTrain)
    yPredLasso = lasso.predict(XTestScaled)

    # ------------------------
    # Gradient Boosting (tuned)
    # ------------------------
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb.fit(XTrain, yTrain)
    yPredGb = gb.predict(XTest)

    # ------------------------
    # Hybrid (weighted)
    # ------------------------
    alpha = 0.05
    yPredHybrid = alpha * yPredRidge + (1 - alpha) * yPredGb

    # ------------------------
    # Store RMSE
    # ------------------------
    ridgeRmseList.append(rmse(yTest, yPredRidge))
    lassoRmseList.append(rmse(yTest, yPredLasso))
    gbRmseList.append(rmse(yTest, yPredGb))
    hybridRmseList.append(rmse(yTest, yPredHybrid))

    # Store for residual plot
    allYTest.extend(np.expm1(yTest))
    allRidgePred.extend(np.expm1(yPredRidge))
    allHybridPred.extend(np.expm1(yPredHybrid))

# ========================
# Results
# ========================
print("\n=== Final Results ===")
print("Ridge Avg:", np.mean(ridgeRmseList))
print("Lasso Avg:", np.mean(lassoRmseList))
print("GB Avg:", np.mean(gbRmseList))
print("Hybrid Avg:", np.mean(hybridRmseList))

# ========================
# Graph 1: Model Comparison
# ========================
models = ["Ridge", "Lasso", "GB", "Hybrid"]
scores = [
    np.mean(ridgeRmseList),
    np.mean(lassoRmseList),
    np.mean(gbRmseList),
    np.mean(hybridRmseList)
]

plt.figure()
plt.bar(models, scores)
plt.ylabel("RMSE")
plt.title("Model Comparison (Improved)")
plt.show()

# ========================
# Graph 2: RMSE per Fold
# ========================
plt.figure()
plt.plot(ridgeRmseList, marker='o', label="Ridge")
plt.plot(lassoRmseList, marker='o', label="Lasso")
plt.plot(gbRmseList, marker='o', label="GB")
plt.plot(hybridRmseList, marker='o', label="Hybrid")

plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("RMSE Across Folds")
plt.legend()
plt.show()

# ========================
# Graph 3: Residual Comparison
# ========================
plt.figure()
plt.scatter(allYTest, np.array(allYTest) - np.array(allRidgePred),
            alpha=0.3, label="Ridge")
plt.scatter(allYTest, np.array(allYTest) - np.array(allHybridPred),
            alpha=0.3, label="Hybrid")

plt.xlabel("True Price")
plt.ylabel("Residual")
plt.title("Residual Comparison")
plt.legend()
plt.show()