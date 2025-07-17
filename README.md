# 🚢 Titanic Survival Prediction

Predict survival on the Titanic using logistic regression and machine learning in Python.

## 🔧 Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

## 📂 Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- File used: `train.csv`

## 📋 Features Used
- Pclass (Ticket class)
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Fare
- Embarked (Port of Embarkation)

## 🧹 Data Preprocessing
- Filled missing values in `Age` and `Embarked`
- Dropped `Cabin`, `Name`, `Ticket`, `PassengerId`
- Label encoded `Sex` and `Embarked`

## 📈 Model
- Logistic Regression
- Train-Test Split: 80/20
- Evaluation:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

## 📊 Results
- Model Accuracy: ~78–82%
- Insights:
  - Females had higher survival rates
  - Higher-class passengers had better survival chances

## 📉 Visualization Example
```python
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Survival by Gender')
plt.show()
