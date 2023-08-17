import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("C:/Users/DELL/Downloads/Dataset.csv")
df.head()

df.dtypes
df.columns

# Handle Missing Values
# For numerical features, fill missing values with the mean
numeric_features = ['age', 'bmi']
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())


# Encode Categorical Features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
encoder = LabelEncoder()
categorical_features = ['ethnicity', 'gender']
for col in categorical_features:
df[col] = encoder.fit_transform(df[col].astype(str))

# Interactive Visualizations using Plotly Express
# Scatter plot to explore relationships between age and BMI
scatter_age_bmi = px.scatter(df, x='age', y='bmi', color='sepsis_label',
title='Relationship between Age, BMI, and Sepsis')
scatter_age_bmi.show()

# Parallel coordinates plot to visualize multiple features at once
parallel_plot = px.parallel_coordinates(df, dimensions=['age', 'bmi', 'ethnicity', 'gender'],
color='sepsis_label',
labels={'age': 'Age', 'bmi': 'BMI', 'ethnicity': 'Ethnicity', 'gender': 'Gender'},
title='Parallel Coordinates Plot of Features and Sepsis')
parallel_plot.show()

# Box plot to compare the distribution of age among sepsis and non-sepsis cases
box_age = px.box(df, x='sepsis_label', y='age', title='Age Distribution among Sepsis and Non-Sepsis Cases')
box_age.show()

# Histogram to explore the distribution of BMI among sepsis and non-sepsis cases
hist_bmi = px.histogram(df, x='bmi', color='sepsis_label', title='BMI Distribution among Sepsis and Non-Sepsis Cases')
hist_bmi.show()

# Define sepsis label based on diagnoses or other relevant indicators
df['sepsis_label'] = (df['aids'] == 1) | (df['cirrhosis'] == 1) | (df['diabetes_mellitus'] == 1)

# Select relevant features for sepsis detection
selected_features = ['age', 'bmi', 'sepsis_label'] # Update with the correct feature names
data_selected = df[selected_features]

# Splitting data into features (X) and target (y)
X = data_selected.drop(['sepsis_label'], axis=1)
y = data_selected['sepsis_label']

# Splitting data into features (X) and target (y)
X = data_selected.drop(['sepsis_label'], axis=1)
y = data_selected['sepsis_label']

# Train and Evaluate Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# Train and Evaluate Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Train and Evaluate K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)
print("KNN Classification Report:\n", classification_report(y_test, knn_predictions))

# Create a DataFrame to store model names and accuracies
import plotly.express as px
import numpy as np 
models = ['Random Forest', 'SVM', 'KNN']
accuracies = [rf_accuracy, svm_accuracy, knn_accuracy]
std_devs = [np.std(rf_predictions), np.std(svm_predictions), np.std(knn_predictions)]
model_data = pd.DataFrame({'Model': models, 'Accuracy': accuracies, 'Std Dev': std_devs})
# Plot Advanced-Level Graph using Plotly
fig = px.bar(model_data, x='Model', y='Accuracy', error_y='Std Dev',
 labels={'Accuracy': 'Accuracy Score', 'Std Dev': 'Standard Deviation'},
 title='Model Accuracy Comparison', color_discrete_sequence=['royalblue'])
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.show()

import time # For simulating new data arrival
# Online Learning Simulation
for _ in range(10): # Simulate 10 new data points
 # Simulate new data point
 new_data_point = {
 'age': np.random.randint(18, 90),
 'bmi': np.random.uniform(15, 40),
 'ethnicity': np.random.choice(data['ethnicity'].unique()),
 'gender': np.random.choice(data['gender'].unique()),
 'sepsis_label': np.random.choice([True, False])
 }
 # Update models with the new data point
 new_data_df = pd.DataFrame([new_data_point])
 X_new = scaler.transform(new_data_df[['age', 'bmi']])
 y_new = new_data_df['sepsis_label'].values
 # Ensure that both classes are represented in the data
 if len(np.unique(y_new)) == 2:
 # Update RandomForest model with warm start
 rf_model.n_estimators += 1
 rf_model.fit(X_new, y_new)
 # Update SVM and KNN models with the new data point
 svm_model.fit(X_new, y_new)
 knn_model.fit(X_new, y_new)
 # Print updated model accuracies
 print("Random Forest Accuracy:", rf_model.score(X_test, y_test))
 print("SVM Accuracy:", svm_model.score(X_test, y_test))
 print("KNN Accuracy:", knn_model.score(X_test, y_test))
 # Pause to simulate data arrival interval
 time.sleep(3) # Wait for 3 seconds before simulating the next data point

# Perform k-fold cross-validation
from sklearn.model_selection import cross_val_score
k = 5 # Number of folds
cv_scores = cross_val_score(rf_model, X, y, cv=k, scoring='accuracy')
# Print cross-validation scores for each fold
for fold, score in enumerate(cv_scores, start=1):
 print(f"Fold {fold}: Accuracy = {score:.4f}")
# Calculate and print the mean and standard deviation of cross-validation scores
mean_score = np.mean(cv_scores)
std_dev_score = np.std(cv_scores)
print(f"Mean Accuracy: {mean_score:.4f}")
print(f"Standard Deviation: {std_dev_score:.4f}")

# Perform k-fold cross-validation
from sklearn.model_selection import cross_val_score
k = 5 # Number of folds
cv_scores = cross_val_score(rf_model, X, y, cv=k, scoring='accuracy')
# Print cross-validation scores for each fold
for fold, score in enumerate(cv_scores, start=1):
 print(f"Fold {fold}: Accuracy = {score:.4f}")
# Calculate and print the mean and standard deviation of cross-validation scores
mean_score = np.mean(cv_scores)
std_dev_score = np.std(cv_scores)
print(f"Mean Accuracy: {mean_score:.4f}")
print(f"Standard Deviation: {std_dev_score:.4f}")

# Perform k-fold cross-validation for the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_cv_scores = cross_val_score(knn_model, X, y, cv=k, scoring='accuracy')

# Calculate and print the mean and standard deviation of cross-validation scores for SVM
svm_mean_score = np.mean(svm_cv_scores)
svm_std_dev_score = np.std(svm_cv_scores)
print("SVM:")
print(f"Mean Accuracy: {svm_mean_score:.4f}")
print(f"Standard Deviation: {svm_std_dev_score:.4f}")
# Calculate and print the mean and standard deviation of cross-validation scores for KNN
knn_mean_score = np.mean(knn_cv_scores)
knn_std_dev_score = np.std(knn_cv_scores)
print("KNN:")
print(f"Mean Accuracy: {knn_mean_score:.4f}")
print(f"Standard Deviation: {knn_std_dev_score:.4f}")

# Initialize individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Initialize the VotingClassifier with the individual models
ensemble_model = VotingClassifier(
 estimators=[('rf', rf_model), ('svm', svm_model), ('knn', knn_model)],
 voting='soft' # Use 'soft' voting for probability-based predictions
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model
ensemble_accuracy = ensemble_model.score(X_test, y_test)
print("Ensemble Model Accuracy:", ensemble_accuracy)








  

