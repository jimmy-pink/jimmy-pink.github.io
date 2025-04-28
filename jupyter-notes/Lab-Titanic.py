import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ╔════════════════════════════════════════╗
# ║ Step 1: Data Collection and Loading    ║
# ╚════════════════════════════════════════╝
# First, we collect the dataset from various sources (CSV, database, API, etc.).
# Make sure the data is in a consistent format and loaded into memory properly.
# Ensure that the dataset contains all the necessary features and labels.
# Check for any corruption or inconsistencies in the data.
titanic_ds = sns.load_dataset("titanic")
print(titanic_ds.shape)
print(titanic_ds.count())
# ╔════════════════════════════════════════╗
# ║ Step 2: Data Cleaning and Preparation  ║
# ╚════════════════════════════════════════╝
# The next step is to clean the data. We will check for missing or null values.
# Missing data will be either filled with an appropriate value or dropped.
# Ensure that the features are correctly formatted (e.g., numeric, categorical).
# If necessary, encode categorical variables into numerical values.
# Also, handle any outliers or erroneous data that could affect model performance.

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic_ds[features]
y = titanic_ds[target]
print(y.value_counts())
print(titanic_ds['deck'].value_counts()) # 只有两百个有数据
# ╔════════════════════════════════════════╗
# ║ Step 3: Feature Engineering             ║
# ╚════════════════════════════════════════╝
# In this step, we will extract and create new features from the raw data.
# This could involve creating interaction terms, aggregating data, or transforming features.
# Select relevant features that are likely to help the model learn better.
# You may also scale or normalize the features to ensure they are on a similar range.

# ╔════════════════════════════════════════╗
# ║ Step 4: Train-Test Split               ║
# ╚════════════════════════════════════════╝
# Now, we will split the dataset into training and testing sets.
# Typically, 80% of the data is used for training and 20% for testing.
# Ensure that the split is random but that the classes are well-represented in both sets.
# This helps to avoid bias and ensures a fair evaluation of the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # 使用中位数补充缺失值
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 用最频繁的值补充缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 转成n-1个不重复0-1特征
])


# ╔════════════════════════════════════════╗
# ║ Step 5: Model Selection               ║
# ╚════════════════════════════════════════╝
# Select an appropriate machine learning model for the task at hand.
# This could be a classification algorithm (e.g., Random Forest, SVM) or regression model.
# Consider the size of the dataset and the problem's complexity when choosing the model.
# Tune hyperparameters to find the optimal configuration for the model.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# ╔════════════════════════════════════════╗
# ║ Step 6: Model Training                ║
# ╚════════════════════════════════════════╝
# Now, we will train the selected model using the training data.
# Feed the features and labels to the model, and let it learn the patterns in the data.
# Keep track of the training progress, and ensure that the model is fitting well to the data.
# If necessary, adjust hyperparameters or use techniques like cross-validation to improve training.
cv = StratifiedKFold(n_splits=5, shuffle=True)
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

# ╔════════════════════════════════════════╗
# ║ Step 7: Model Evaluation              ║
# ╚════════════════════════════════════════╝
# Once the model is trained, it's time to evaluate its performance.
# We will use the testing set to make predictions and compare them to the actual labels.
# Calculate various metrics like accuracy, precision, recall, F1-score, etc.
# Make sure that the model generalizes well to new, unseen data and is not overfitting.
y_pred = model. predict(X_test)
print(classification_report(y_test, y_pred))

# ╔════════════════════════════════════════╗
# ║ Step 8: Hyperparameter Tuning         ║
# ╚════════════════════════════════════════╝
# After evaluating the model, you may notice that it can be improved.
# Use techniques like Grid Search or Random Search to tune the hyperparameters.
# Try different values for parameters such as learning rate, number of trees, etc.
# Evaluate the model again after tuning to see if performance improves.
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_importances = model.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

# Print test score
test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")