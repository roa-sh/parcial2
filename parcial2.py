import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA 
from sklearn.tree import export_graphviz



#se carga el dataset previamente limpiado en la actividad de RapidMiner
dataframe = pd.read_csv("./student-mat.csv",  sep=';')
dataframe =  dataframe.dropna()
feature_cols = dataframe.columns

dataframe["aprobacion"] = np.where(dataframe['G3'] >= 12 , 'aprobado','no aprobado')
print(dataframe[["G3","aprobacion"]])


# Punto 5 

X = dataframe[feature_cols] # Features
y = dataframe.aprobacion # Target variable
# Split dataset into training set and test set
#random_state generador de numero aleatorios (semilla)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

dot_data = export_graphviz(clf, out_file=None, filled=True, feature_names = feature_cols, class_names=['aprobado','no aprobado'])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('tree2.pdf')

