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

# Log transform
y = np.log1p(df["SalePrice"])
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

# ========================
# Preprocessing
# ========================
numCols = X.select_dtypes(include=['number']).columns
catCols = X.select_dtypes(include=['object']).columns

X[numCols] = X[numCols].fillna(X[numCols].median())
X[catCols] = X[catCols].fillna("Missing")

qualMap = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Missing":0}
for col in ["KitchenQual","ExterQual"]:
    if col in X.columns:
        X[col] = X[col].map(qualMap)

X = pd.get_dummies(X)

# ========================
# K-Fold Setup
# ========================
kFold = KFold(n_splits=5, shuffle=True, random_state=42)

ridgeList, lassoList, gbList, hybridList, stackList = [], [], [], [], []

allYTest, allGbPred = [], []

def rmse(yTrue, yPred):
    return np.sqrt(mean_squared_error(np.expm1(yTrue), np.expm1(yPred)))

# ========================
# Training Loop
# ========================
for trainIdx, testIdx in kFold.split(X):

    XTrain, XTest = X.iloc[trainIdx], X.iloc[testIdx]
    yTrain, yTest = y.iloc[trainIdx], y.iloc[testIdx]

    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(XTrainScaled, yTrain)
    yPredRidge = ridge.predict(XTestScaled)

    # Lasso
    lasso = Lasso(alpha=0.0005, max_iter=10000)
    lasso.fit(XTrainScaled, yTrain)
    yPredLasso = lasso.predict(XTestScaled)

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb.fit(XTrain, yTrain)
    yPredGb = gb.predict(XTest)

    # Hybrid
    alpha = 0.05
    yPredHybrid = alpha*yPredRidge + (1-alpha)*yPredGb

    # Stacking
    stackTrain = np.vstack([ridge.predict(XTrainScaled), gb.predict(XTrain)]).T
    stackTest = np.vstack([yPredRidge, yPredGb]).T

    meta = Ridge(alpha=1.0)
    meta.fit(stackTrain, yTrain)
    yPredStack = meta.predict(stackTest)

    # Store RMSE
    ridgeList.append(rmse(yTest, yPredRidge))
    lassoList.append(rmse(yTest, yPredLasso))
    gbList.append(rmse(yTest, yPredGb))
    hybridList.append(rmse(yTest, yPredHybrid))
    stackList.append(rmse(yTest, yPredStack))

    # store for graphs
    allYTest.extend(np.expm1(yTest))
    allGbPred.extend(np.expm1(yPredGb))

# ========================
# Results
# ========================
print("\n=== FINAL RESULTS ===")
print("Ridge:", np.mean(ridgeList))
print("Lasso:", np.mean(lassoList))
print("GB:", np.mean(gbList))
print("Hybrid:", np.mean(hybridList))
print("Stack:", np.mean(stackList))

# ========================
# Graph 1: Model Comparison
# ========================
plt.figure()
plt.bar(["Ridge","Lasso","GB","Hybrid","Stack"],
        [np.mean(ridgeList),np.mean(lassoList),
         np.mean(gbList),np.mean(hybridList),np.mean(stackList)])
plt.title("Model Comparison")
plt.ylabel("RMSE")
plt.show()

# ========================
# Graph 2: RMSE per Fold
# ========================
plt.figure()
plt.plot(ridgeList, label="Ridge")
plt.plot(lassoList, label="Lasso")
plt.plot(gbList, label="GB")
plt.plot(hybridList, label="Hybrid")
plt.plot(stackList, label="Stack")
plt.legend()
plt.title("RMSE Across Folds")
plt.show()

# ========================
# Graph 3: Actual vs Predicted
# ========================
plt.figure()
plt.scatter(allYTest, allGbPred, alpha=0.5)
plt.plot([0,max(allYTest)], [0,max(allYTest)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (GB)")
plt.show()

# ========================
# Graph 4: Residual Distribution
# ========================
residuals = np.array(allYTest) - np.array(allGbPred)

plt.figure()
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.show()

# ========================
# Graph 5: Error vs Price
# ========================
errors = abs(residuals)

plt.figure()
plt.scatter(allYTest, errors, alpha=0.5)
plt.xlabel("Price")
plt.ylabel("Error")
plt.title("Error vs Price")
plt.show()

# ========================
# Graph 6: Feature Importance
# ========================
importances = gb.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top Features")
plt.show()