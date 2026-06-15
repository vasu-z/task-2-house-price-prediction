# House Price Prediction – Task 2

## Overview

This project develops an advanced machine learning regression pipeline capable of predicting residential house prices based on various structural, spatial, geometric, and seasonal characteristics using the Ames Housing dataset.

## Technologies Used

* Python
* Kaggle Notebook Environment
* Pandas
* NumPy
* SciPy (Stats & Special Packages)
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn

## Machine Learning Models

* **Base Learners:**
* Ridge Regression
* Lasso Regression
* Support Vector Regression (SVR)
* Gradient Boosting Regressor
* XGBoost Regressor
* LightGBM Regressor


* **Meta Learner:**
* Stacking Ensemble Regressor (with Ridge Final Estimator)



## Project Workflow

1. **Data Collection & Loading:** Loaded training and test datasets into Pandas dataframes and isolated unique tracking identifiers.
2. **Exploratory Data Analysis (EDA):** Performed spatial and numerical target relationship checks, mapped missing value frequencies, and profiled structural anomalies.
3. **Outlier Mitigation:** Cleaned the dataset by dropping extreme anomalies where ground living area (`GrLivArea`) exceeded 4,500 sq. ft.
4. **Target Distribution Optimization:** Fixed target column skewness by applying a logarithmic transformation ($y = \text{np.log1p}(y)$) to achieve a standard normal bell curve.
5. **Context-Aware Imputation:** Handled missing properties using localized neighborhood-grouped modes for categorical items, neighborhood-grouped medians for dimensions, and zero-fills for implicit zero properties.
6. **Feature Engineering:** Built dense composite tracking signals (`TotalArea`, `TotalBaths`, `TotalPorch`), extracted boolean presence markers for premium luxuries, and engineered continuous cyclical time metrics (`sin`/`cos`) for months sold.
7. **High-Dimensional Scaling & Encoding:** Applied automated Box-Cox parameter sweeps (`boxcox_normmax`) and log fallbacks to skewed columns, one-hot encoded all textual factors, and transformed numeric features with `RobustScaler`.
8. **Hyperparameter Tuning:** Conducted optimized cross-validation search queries over all base-model pipelines using a 5-fold Stratified `KFold` distribution pattern.
9. **Ensemble Stacking:** Fused individual tuned predictors into a unified native `StackingRegressor` to minimize structural overfitting.
10. **Prediction & Inverse Scaling:** Calculated blind test set values and inverted log predictions via `np.expm1()` back into actual home dollar figures.

## Outputs

* Optimized Model Parameters (RandomizedSearchCV)
* Training vs Test Evaluation Curves
* 5-Fold Cross-Validation Scores
* Final Exported Prediction File (`my_prediction_ensemble.csv`)

## Result

The stacking ensemble model successfully optimized structural regression targets, drastically decreasing prediction error and improving test-set generalization over individual base-regression models.
