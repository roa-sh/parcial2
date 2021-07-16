import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
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


#se carga el dataset previamente limpiado en la actividad de RapidMiner
dataframe = pd.read_csv("./student-mat.csv",  sep=';')
dataframe =  dataframe.dropna()
print(dataframe)

print(dataframe['Medu'].min())

