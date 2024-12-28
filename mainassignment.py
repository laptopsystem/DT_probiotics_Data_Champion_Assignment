import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



# Load dataset
url = "https://github.com/laptopsystem/DT_probiotics_Data_Champion_Assignment/blob/main/companies.csv"
df = pd.read_csv(url)

# Cleaning ,preprocess data
df_cleaned = df.drop_duplicates()
df_cleaned['Revenue'] = df_cleaned['Revenue'].fillna(df_cleaned['Revenue'].mean())
df_cleaned['Growth_Rate'] = df_cleaned['Growth_Rate'].fillna(df_cleaned['Growth_Rate'].mean())
df_cleaned['Company_ID'] = df_cleaned['Company_ID'].fillna(-1)
df_cleaned['Prospect'] = df_cleaned['Growth_Rate'].apply(lambda x: 'Yes' if x > 20 else 'No')

# app layout
st.title('DT Probiotics Data Champion - Automated Dashboard')
st.write('This is a dashboard to analyze company data and predict prospects.')

# Links for Dataset 
st.write("### 1. Dataset Link: [Link](https://github.com/laptopsystem/DT_probiotics_Data_Champion_Assignment/blob/main/companies.csv)")
st.write("### 3. Github: [Link](https://github.com/laptopsystem/DT_probiotics_Data_Champion_Assignment)")
# Data Display Section
st.subheader('Data Overview')
st.write(df_cleaned.head())  #  5 rows

# Scatter plot of Revenue vs Growth Rate
st.subheader('Revenue vs Growth Rate')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='Revenue', y='Growth_Rate', hue='Prospect', palette='coolwarm', ax=ax)
st.pyplot(fig)

# Machine Learning Section
st.subheader('Model Training and Prediction')

# Prepare features and labels
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
X = df_encoded.drop(columns=['Company_ID', 'Company_Name', 'Prospect'], errors='ignore')
y = df_cleaned['Prospect'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_y_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_y_pred)

# Display Logistic Regression Accuracy
st.write(f"Logistic Regression Model Accuracy: {log_accuracy:.2f}")

# Model Comparison (Other Models)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"{model_name} Accuracy: {accuracy:.2f}")
    
    # Additional Model Metrics: Confusion Matrix and Classification Report
    st.subheader(f"{model_name} - Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader(f"{model_name} - Classification Report")
    clf_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    st.text(clf_report)

# Correlation Matrix Heatmap
st.subheader('Correlation Matrix')
df_numeric = df_cleaned.select_dtypes(include=[np.number])  # Only numeric columns
correlation_matrix = df_numeric.corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Distribution of prospects by industry
st.subheader('Prospect Distribution by Industry')
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Industry', hue='Prospect')
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution of prospects by location
st.subheader('Prospect Distribution by Location')
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Location', hue='Prospect')
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution of Revenue and Growth Rate
st.subheader('Revenue and Growth Rate Distribution')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df_cleaned['Revenue'], kde=True, ax=ax1, color='skyblue')
ax1.set_title('Revenue Distribution')

sns.histplot(df_cleaned['Growth_Rate'], kde=True, ax=ax2, color='green')
ax2.set_title('Growth Rate Distribution')

st.pyplot(fig)

# Download the processed data (if needed)
st.subheader('Download Processed Data')
df_cleaned.to_csv("Processed_Prospect_Companies.csv", index=False)
st.download_button('Download Processed Data', data=open('Processed_Prospect_Companies.csv', 'rb').read(), file_name="Processed_Prospect_Companies.csv")
