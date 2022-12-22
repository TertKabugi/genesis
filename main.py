import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import pyplot
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('crop.csv', header=0)
df = df.dropna()
print(df.shape)
print()
print(df)


print(df["crop"].unique())
print(df["crop"].value_counts())

summary = pd.pivot_table(df, index="crop", aggfunc="mean")
print(summary)

#data exploration
newSummary = summary.reset_index()
print(newSummary)

# graph where y-axis = N
plt.figure(figsize=(15,6))
sns.barplot(y = 'N', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig1 = px.bar(newSummary, x='crop', y='N')
fig1.show()

# graph where y-axis = P
plt.figure(figsize=(15,6))
sns.barplot(y = 'P', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig2 = px.bar(newSummary, x='crop', y='P')
fig2.show()

# graph where y-axis = K
plt.figure(figsize=(15,6))
sns.barplot(y = 'K', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig3= px.bar(newSummary, x='crop', y='K')
fig3.show()

# graph where y-axis = temperature
plt.figure(figsize=(15,6))
sns.barplot(y = 'temperature', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig4 = px.bar(newSummary, x='crop', y='temperature')
fig4.show()

# graph where y-axis = humidity
plt.figure(figsize=(15,6))
sns.barplot(y = 'humidity', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig5 = px.bar(newSummary, x='crop', y='humidity')
fig5.show()

# graph where y-axis = pH
plt.figure(figsize=(15,6))
sns.barplot(y = 'ph', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig6 = px.bar(newSummary, x='crop', y='ph')
fig6.show()

# graph where y-axis = rainfall
plt.figure(figsize=(15,6))
sns.barplot(y = 'rainfall', x = 'crop', data=newSummary, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()
fig7 = px.bar(newSummary, x='crop', y='rainfall')
fig7.show()


# Data Pre-processing(cleaning data)
label_encoder = preprocessing.LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['crop'])
print(df["crop"].unique())

# Features(x): [N, P, K, temperature, humidity, ph, rainfall]
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop']

# Training, Validating and Testing Sets
X_train, X_val_, y_train, y_val_ = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_val_, y_val_, test_size = 0.5, random_state = 42)

print("Shape of the X Train :", X_train.shape)
print(X_train)
print("Shape of the y Train :", y_train.shape)
print(y_train)
print("Shape of the X val :", X_val.shape)
print(X_val)
print("Shape of the y val :", y_val.shape)
print(y_val)
print("Shape of the X test :", X_test.shape)
print(X_test)
print("Shape of the y test :", y_test.shape)
print(y_test)


# Feature Selection
fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X_train, y_train)
feature_names=X.columns
for i in range(len(fs.scores_)):
	print('Importance of '+feature_names[i]+' is %f' % (fs.scores_[i]))

# graph showing feature importance
plt.rcParams["figure.figsize"] = (8,8)
pyplot.bar([i for i in X.columns], fs.scores_)
pyplot.show()


# Machine learning models and accuracy rates
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

# model used
model = LogisticRegression()

# fit the model
model.fit(X_train, y_train)
print(model)

LG_accuracy = model.score(X_test,y_test)
print("On predicting the test set, the model gets an accuracy of: ", LG_accuracy)

# Logistic Regression
LG = LogisticRegression()
LG.fit(X_train , y_train)

# Saving model to disk
pickle.dump(LG, open("model.pkl", "wb"))

