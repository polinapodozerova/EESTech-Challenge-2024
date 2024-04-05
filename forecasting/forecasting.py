import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data, preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

def oil(x):
    if type(x)==float:
        return x
    else:
        return int(str(x).split(',')[0])
    
def change_to_num(dataset):
    dataset['КПП. Температура масла'] = pd.to_numeric(dataset['КПП. Температура масла'])
    dataset['Давл.масла двиг.,кПа'] = pd.to_numeric(dataset['Давл.масла двиг.,кПа'])
    dataset['Темп.масла двиг.,°С'] = pd.to_numeric(dataset['Темп.масла двиг.,°С'].apply(lambda x: oil(x)))
    dataset['КПП. Давление масла в системе смазки'] = pd.to_numeric(dataset['КПП. Давление масла в системе смазки'])
    dataset['Скорость'] = pd.to_numeric(dataset['Скорость'])
    dataset['ДВС. Давление смазки'] = pd.to_numeric(dataset['ДВС. Давление смазки'])
    dataset['ДВС. Температура охлаждающей жидкости'] = pd.to_numeric(dataset['ДВС. Температура охлаждающей жидкости'])
    dataset['Давление в пневмостистеме (spn46), кПа'] = pd.to_numeric(dataset['Давление в пневмостистеме (spn46), кПа'])
    dataset['Уровень топлива % (spn96)'] = pd.to_numeric(dataset['Уровень топлива % (spn96)'])
    dataset['Электросистема. Напряжение'] = pd.to_numeric(dataset['Электросистема. Напряжение'])
    dataset['ДВС. Частота вращения коленчатого вала'] = pd.to_numeric(dataset['ДВС. Частота вращения коленчатого вала'])
    return dataset

problem_data = pd.read_csv('dataset._problems.csv', sep=';').iloc[:1000000]
normal_data = pd.read_csv('dataset._normal.csv', sep=';')
# example_input = pd.read_csv('023T0569\log(336804727)[26-03-2024_16-11-37] 01.07-01.08.csv', sep=';').replace('-273,000', -273).replace('33,000', 33).iloc[390]
problem_data['Дата и время'] = pd.to_datetime(problem_data['Дата и время'], format="mixed")
problem_data = preprocess_data(problem_data)
normal_data = preprocess_data(normal_data)
# try:
#     example_input = preprocess_data(pd.DataFrame(example_input).T);
# except:
#     example_input = preprocess_data(pd.DataFrame(example_input));

problem_data['target'] = np.zeros(len(problem_data)) + 1
normal_data['target'] = np.zeros(len(normal_data))

data = change_to_num(pd.concat([problem_data, normal_data], ignore_index=True, axis=0).sample(frac=1.0, random_state=42).drop(columns=['Дата и время']))

X = data.drop('target', axis=1)
y = data['target']

problem_data['target'] = np.zeros(len(problem_data)) + 1
normal_data['target'] = np.zeros(len(normal_data))

data = change_to_num(pd.concat([problem_data, normal_data], ignore_index=True, axis=0).sample(frac=1.0, random_state=42).drop(columns=['Дата и время']))

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc}")

joblib.dump(model, 'model.pkl')