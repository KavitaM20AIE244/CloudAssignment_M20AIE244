import streamlit as str 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

str.title('Welcome Kavita Joshi')

str.write("""
Classifier and Datasets

""")

dset_name = str.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

str.write(f"## {dset_name} Dataset")

classifier_name = str.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_data(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_data(dset_name)
str.write('Shape of dataset:', X.shape)
str.write('number of classes:', len(np.unique(y)))

def add_parameter(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = str.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = str.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = str.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = str.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter(classifier_name)

def get_classifier(classifier_name, params):
    classifier = None
    if classifier_name == 'SVM':
        classifier = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        classifier = classifier = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return classifier

classifier = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

str.write(f'Classifier = {classifier_name}')
str.write(f'Accuracy =', accuracy)

#### PLOT DATASET ####

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar()

#plt.show()
str.pyplot(fig)
