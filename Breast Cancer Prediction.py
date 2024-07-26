import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants
NUM_FOLDS = 10
TEST_SIZE = 0.20
RANDOM_STATE = 21

# Load the dataset
try:
    data = pd.read_csv(r'C:\Users\xpert\Downloads\data.csv', index_col=False)
except Exception as e:
    print("Error loading dataset: ", str(e))
    exit()

# Print the first 5 rows of the dataset
print(data.head(5))

# Print the shape of the dataset
print(data.shape)

# Print the summary statistics of the dataset
print(data.describe())

# Convert the diagnosis column to binary values
data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')

# Set the id column as the index
data = data.set_index('id')

# Remove the Unnamed: 32 column
if 'Unnamed: 32' in data.columns:
    del data['Unnamed: 32']

# Print the count of each diagnosis class
print(data.groupby('diagnosis').size())

# Plot the density plots for each feature
data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()

# Plot the correlation matrix
fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = plt.get_cmap('jet', 30)
cax = ax1.imshow(data.corr(), interpolation="none", cmap=cmap)
ax1.grid(True)
plt.title('Breast Cancer Attributes Correlation')
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()

# Split the dataset into features (X) and target (Y)
Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Define the models to be compared
models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))

# Compare the performance of each model
results = []
names = []
for name, model in models_list:
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end - start))

# Visualize the results
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Train the best model on the full training dataset and evaluate on the test set
best_model = SVC()
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_test)

# Print the classification report
print(classification_report(Y_test, predictions))

# Print the confusion matrix
print(confusion_matrix(Y_test, predictions))

# Print the accuracy score
print("Accuracy: ", accuracy_score(Y_test, predictions))
