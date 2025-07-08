from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error,classification_report
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('/content/drive/MyDrive/CSE-303_Project/insurance_csv.csv')

df

df.head()

df.tail()

df.isnull().sum()

df.info()

# Display duplicate rows
print("Duplicate rows:")
print(df[df.duplicated()])

# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()

# Verify the number of rows after dropping duplicates
print(f"Number of rows after dropping duplicates: {len(df_no_duplicates)}")

df_copy = df.copy()
df['sex'].value_counts()

# Define the mapping dictionary
sex_mapping = {
    'male': 1,
    'female': 2,
}

# Apply the mapping
df['sex'] = df['sex'].map(sex_mapping)
df

df['smoker'].value_counts()

# Define the mapping dictionary
smoker = {"yes":1,"no":0}

# Apply the mapping
df['smoker'] = df['smoker'].map(smoker)
df

df['region'].value_counts()

# Define the mapping dictionary
region = {"southwest":1,"southeast":2,"northwest":3,"northeast":4}

# Apply the mapping
df['region'] = df['region'].map(region)
df

correlation = df.corr()
correlation

#Correation matrix
plt.figure(figsize=(12,10), dpi=77)
sns.heatmap(correlation, linecolor='white',linewidths=0.1, annot=True)
plt.title('Correlation Matrix'.upper(), size=19, pad=13)
plt.xlabel('Insurance Data')
plt.ylabel('Insurance Data')
plt.show()

#Bar plot
sns.barplot(x='region', y='charges', data=df)
plt.title('Average Medical Cost by Region')
plt.xlabel('Region')
plt.ylabel('Average Charges')
plt.show()

#Line plot
plt.figure(figsize=(10, 8))
sns.lineplot(x='age', y='children', data=df,hue='smoker')
plt.title("Age vs. Children")
plt.xlabel("Age")
plt.ylabel("Number of Children")
plt.show()

#Count plot
sns.countplot(x='smoker', data=df)
plt.title('Number of Smokers')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Smoker', 'Smoker'])
plt.show()

#Histrogram
plt.figure(figsize=(10, 8))
sns.histplot(df['bmi'], kde=True, bins=30)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

#Pie chart

yes = (df['smoker'] == 1).sum()
no = (df['smoker']== 0).sum()
proportions = [yes,no]
print(proportions)
print("Show")
plt.figure(figsize=(12,8), dpi=77)
plt.pie(proportions, data=df, labels= ['Smoker', 'Non-smoker'], explode = (0.05,0), startangle=90, autopct='%1.1f%%', shadow=False)
plt.axis('equal')
plt.title("Smoker Proportion", size=17, pad=13)
plt.show()

#Box plot
plt.figure(figsize=(12, 8), dpi=77)
sns.boxplot(x='children', y='charges', data=df)
plt.title("Children vs. Charges")
plt.xlabel('Number of Children')
plt.ylabel('Charges')
plt.show()

#Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette='viridis')
plt.title("BMI vs Charges")
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.legend(title='Smoker')
plt.show()

X=df.drop(columns=['charges'])
Y=df['charges']

X = X.dropna()
Y = Y[X.index]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

from sklearn.linear_model import  LinearRegression

model= LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report, r2_score
from sklearn.linear_model import LogisticRegression

mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")



