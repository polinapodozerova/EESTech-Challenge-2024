import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data, preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

anomaly_data = pd.read_csv('preprocessed_anomaly_data.csv')
normal_data = pd.read_csv('preprocessed_normal_data.csv')


example_input = pd.read_csv('023T0569\log(336804727)[26-03-2024_16-11-37] 01.07-01.08.csv', sep=';').iloc[193]
try:
    example_input = preprocess_data(pd.DataFrame(example_input).T);
except:
    example_input = preprocess_data(pd.DataFrame(example_input));

anomaly_data['target'] = np.zeros(len(anomaly_data)) + 1
normal_data['target'] = np.zeros(len(normal_data))

data = pd.concat([anomaly_data, normal_data], ignore_index=True, axis=0).sample(frac=1.0, random_state=42).drop(columns=['Unnamed: 0', 'Дата и время'])

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# model.predict(example_input.drop(columns='Дата и время'))
joblib.dump(model, 'anomaly_model.pkl')