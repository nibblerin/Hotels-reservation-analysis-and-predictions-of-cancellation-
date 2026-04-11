The project consists of several key parts:
1. Data exploration and visualization:
- EDA to understand the features, identify trends, and check for patterns in booking cancellations;
- Visualization of different aspects of the data.
2. Providing hypotheses that longer lead time increases the likelihood of a booking being canceled, and the opposite impact of special requests on cancellations.
3. Data preprocessing & engineering:
- Building a preprocessing pipeline using ColumnTransformer. This includes OneHotEncoder for categorical variables and StandardScaler for numerical features to ensure optimal performance for Logistic Regression. 
4. Model training and evaluation (Logistic Regression and CatBoost).
5. Using permutation importance to identify the most important features affecting cancellations.
6. Visualizing important features with Partial Dependence Plots and SHAP for specific customer segments.
