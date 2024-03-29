
# MNIST

### Algorithms

#### What are one key strength and weakness for the following algorithm: Decision Tree, K Neighbor, SVC, SGD

DecisionTreeClassifier is easy to interpret and useful for nonlinear data relationship. However, it is prone to overfitting and the model capture the noise in the data. 

KNeighborsClassifier is effective on basic classification problem. Yet, the performance significantly degrades with high dimensional data that uses more computational resources. 

SVC is useful in capturing complex relationship in data and works well with both linear and non-linear boundaries. However, choosing the right hyperparameters is challenging. 

SGDClassifier is efficient is large scale dataset, however, the classifier is sensitive to feature scaling and require careful tuning of hyperparameters. 

#### Explain if increasing the hyperparameter value will lead to more or less overfitting


For DecisionTreeClassifier, increase in max_depth leads to more overfitting, while increase in min_samples_leaf leads to less overfitting. 

For KNeighborsClassifier, increase in n_neighbors leads to less overfitting while changing the weights indirectly lead to overfitting if there is more weight to nearer neighbors. 

SVC is prone to overfitting when increases C. Furthermore, increasing degree leads to more overfitting as the model becomes more complex. The kernel parameters will also affect the fitting. For instance, high C with 'linear' kernel behaves differently with 'poly' kernel. 

SGDClassifier increases alpha reduces overfitting, while increasing eta0 can affect the complex effect. Too high may lead to optimal solution while too low, it may converge efficiently. Also, increasing max_iter can lead to more overfitting. 

### Imports


```python
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
```

### Load data


```python
mnist_28x28_train = np.load("MNIST/mnist_28x28_train.npy")
mnist_8x8_train = np.load("MNIST/mnist_8x8_train.npy")

mnist_28x28_test = np.load("MNIST/mnist_28x28_test.npy")
mnist_8x8_test = np.load("MNIST/mnist_8x8_test.npy")

#one dimensional array with categorical labels
train_labels = np.load("MNIST/train_labels.npy")
```

###  Data exploration

#### Question 3:



28x28 is expected to perform better than 8x8 since the pixel for 28x28 has a clearer image when plotted compared to 8x8. It makes it easier for classification.


```python
index = 0 
plt.figure(figsize=(10,5))

# Plot MNIST 28x28 image
plt.subplot(1, 2, 1)
plt.imshow(mnist_28x28_train[index], cmap='gray')
plt.title("MNIST 28x28")
plt.axis('off')

# Plot MNIST 8x8 image
plt.subplot(1, 2, 2)
plt.imshow(mnist_8x8_train[index], cmap='gray')
plt.title("MNIST 8x8")
plt.axis('off')

plt.show()
```


![png](output_14_0.png)


#### Question 4:

The minimum accuracy that should be expected from the model is 10.53%, which is a simple baseline that would help us compare against other classifiers. 


```python
from sklearn.dummy import DummyClassifier

# Reshape the data
X = mnist_8x8_train.reshape((mnist_8x8_train.shape[0], -1))
y = train_labels

# Initialize and train the dummy classifier
dummy_clf = clf = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_clf.fit(X, y)
dummy_clf.predict(X)
dummy_clf.score(X, y)
```




    0.10533333333333333



###  Data preparation

#### Question 5:

I have examined the datasets by finding for missing values and its mean and standard deviation. Though there were no missing values, since the mean and standard deviation was not close to 0 and 1 preprocessing the data was needed. 


```python
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#check for missing values
print("Missing values in mnist_28x28_train:", np.isnan(mnist_28x28_train).any())
print("Missing values in mnist_8x8_train:", np.isnan(mnist_8x8_train).any())

#calculate mean and standard deviation:
#reshape data
mnist_28x28_train_reshaped = mnist_28x28_train.reshape(mnist_28x28_train.shape[0], -1)
mnist_8x8_train_reshaped = mnist_8x8_train.reshape(mnist_8x8_train.shape[0], -1)

#mean and std for 28x28:
mean_28 = np.mean(mnist_28x28_train_reshaped)
std_28 = np.std(mnist_28x28_train_reshaped)
print("Mean 28: {}".format(mean_28))
print("Standard Deviation 28: {}".format(std_28))

#mean and std for 8x8:
mean_8 = np.mean(mnist_8x8_train_reshaped)
std_8 = np.std(mnist_8x8_train_reshaped)
print("Mean 8: {}".format(mean_8))
print("Standard Deviation 8: {}".format(std_8))

#preprocessing:
X_28_scaler = preprocessing.StandardScaler().fit(mnist_28x28_train_reshaped)
X_28 = X_28_scaler.transform(mnist_28x28_train_reshaped)

X_8_scaler = preprocessing.StandardScaler().fit(mnist_8x8_train_reshaped)
X_8 = X_8_scaler.transform(mnist_8x8_train_reshaped)
```

    Missing values in mnist_28x28_train: False
    Missing values in mnist_8x8_train: False
    Mean 28: 33.6080537414966
    Standard Deviation 28: 78.89590798753801
    Mean 8: 33.743920833333334
    Standard Deviation 8: 53.99937949606828


#### Question 6:


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

#1. Apply PCA

#reshape data 
mnist_28x28_train_reshaped = mnist_28x28_train.reshape(mnist_28x28_train.shape[0], -1)
mnist_8x8_train_reshaped = mnist_8x8_train.reshape(mnist_8x8_train.shape[0], -1)

#tune the number of components with cumulative explained variance
#find the point diminishing returns
pca = PCA()
pca_28 = PCA().fit(mnist_28x28_train_reshaped)
pca_8 = PCA().fit(mnist_8x8_train_reshaped)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_28.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_8.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Initialize PCA:
pca_28 = PCA(n_components = 50)
pca_28.fit(mnist_28x28_train_reshaped)
pca_28_transform = pca_28.transform(mnist_28x28_train_reshaped)

pca_8 = PCA(n_components = 15)
pca_8.fit(mnist_8x8_train_reshaped)
pca_8_transform = pca_8.transform(mnist_8x8_train_reshaped)
```


![png](output_23_0.png)



![png](output_23_1.png)


### Experiments

#### Question 7:

I have used hold out ot divide the data. Since the dataset is already splitted to X_train (75%) and X_test (25%), I have divided the X_train so that the data is split into 3 parts: train, validation and test. From the X_train data, 20% is reserved for validation. 


```python
from sklearn.model_selection import train_test_split


#split the 28x28 data
X_train_28, X_val_28, y_train_28, y_val_28 = train_test_split(
X_28, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

#split the 8x8 data
X_train_8, X_val_8, y_train_8, y_val_8 = train_test_split(
X_8, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

```

#### Question 8:

To tune the hyperparameters of the SGDClassifier, I have set the learning rate ranges from 0.1, 0.01, 0.001 and number of epochs to 100, 200 and 300. Trying out each of the parameters and fine tuning, I was able to yield the best hyperparameter for learning rate : 0.01 and number of epochs : 200 examining the changes in the final training log loss. 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


learning_rates = 0.01
number_of_epochs = 200
error = []

plt.figure(figsize=(15, 5))
                                           
model = SGDClassifier(warm_start=True, alpha=0.0001, loss='log', learning_rate='constant', eta0=learning_rates, 
                      penalty='l2', random_state=42)
for epoch in range(number_of_epochs):
    model.partial_fit(X_train_8, y_train_8, classes=np.unique(y_train_8).astype(int))
    y_pred = model.predict_proba(X_val_8)
    error.append(log_loss(y_val_8, y_pred, labels=np.unique(y_train_8)))

    # Plot the training curve for each learning rate
plt.plot(range(1, number_of_epochs + 1), error)
plt.title('Training Curve for SGD Classifier')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.show()

print("Final training log loss: {}".format(np.round(error[number_of_epochs -1], 5)))
```


![png](output_30_0.png)


    Final training log loss: 0.52233


#### Question 9:

Cross Validation Score has been used to calculate the accuracy and standard deviation. The mean was used for estimated generalization and the level of uncertainty was used with the standard deviation. 

To ensure the reporducibility of the experiment the random state of 42 is used for all the models.


```python
from sklearn.model_selection import cross_val_score

#list of algorithms
dt_model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3, weights="distance")
svc_model = SVC(C=10, kernel="poly", degree=3, gamma='scale', random_state=42)
sgd_model = SGDClassifier(warm_start=True, alpha=0.0001, loss='log', learning_rate='constant', eta0=0.01, 
                      penalty='l2', random_state=42, max_iter=1000)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)

#cross validation helper function:
def estimate_perf(clf, X, y):
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    return np.mean(scores), np.std(scores)

#find both the accuracy (estimated generalization) and the standard deviation (uncertainty)
perf = {}
for name, model in [("DT", dt_model),
                    ("KNN", knn_model),
                    ("SVC", svc_model),
                    ("SGD", sgd_model),
                    ("Dummy", dummy_model)]:
    mean, std = estimate_perf(model, X_8, train_labels)
    perf[name] = (mean, std)

print("Algorithm\t\tEstimated Generalization")
for name, (accuracy, std) in perf.items():
    print("{}\t\t\t {:.2f} ± {:.2f}".format(name, accuracy * 100, std * 100))
```

    Algorithm		Estimated Generalization
    DT			 76.96 ± 1.50
    KNN			 88.69 ± 1.38
    SGD			 88.14 ± 1.91
    SVC			 92.74 ± 1.21
    Dummy			 10.53 ± 0.07


#### Question 10:

The SVC is the best model due to its high performance on both training and cross validation sets. There was only a minor difference between the accuracy score, suggesting a model without significant overfitting. Therefore, the best algorithm would perform the best. 

The Decision Tress is the worst model with the lowest cross vlaidation accuracy and a high training accuracy, which suggests overfitting. It seems like the data is failing to generalize well, and therefore the worst algorithm performs the worst.


```python
from sklearn.metrics import accuracy_score

# helper function to calculate the accuracy score on training set
def calculate_training_accuracy(model, X_train, y_train):
    model.fit(X_train, y_train)
    pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, pred)
    return accuracy

# Calculate accuracy score on training dataset
for name, model in [("DT", dt_model),
                    ("KNN", knn_model),
                    ("SVC", svc_model),
                    ("SGD", sgd_model),
                    ("Dummy", dummy_model)]:
    train_acc = calculate_training_accuracy(model, X_train_8, y_train_8)
    cv_mean, cv_std = perf[name]
    print("{}:\tTraining Accuracy: {:.2f}%, CV Accuracy: {:.2f} ± {:.2f}%".format(
    name, train_acc * 100, cv_mean * 100, cv_std * 100))

```

    DT:	Training Accuracy: 95.27%, CV Accuracy: 76.96 ± 1.50%
    KNN:	Training Accuracy: 100.00%, CV Accuracy: 88.69 ± 1.38%
    SVC:	Training Accuracy: 98.60%, CV Accuracy: 92.74 ± 1.21%
    SGD:	Training Accuracy: 91.00%, CV Accuracy: 88.14 ± 1.91%
    Dummy:	Training Accuracy: 10.53%, CV Accuracy: 10.53 ± 0.07%


#### Question 11:

To systematically tune the hyperparameters, hyperparameter ranges that are frequently used was chosen. Then with the optimized hyperparameters, I did fine tuning once more. Then GridSearchCV and KFold was used to test every combination of the parameters, and the random_state and n_splits were set constant to ensure reproducibility and validation accuracy.  

The search for the most optimal hyperparameters, there is a trade-off between computational time and performance. For instance, relationship between 'alpha' can require different learning rate for optimal convergence and this will increase the computational cost. 

Each kernel might require a different C value for optimal performance, and yet there needs to be a trade - off with the computational cost and time. 


```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

random_state = 42
n_splits = 10

models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=1),
    "SVC": SVC(random_state=random_state, gamma='scale'),
    "LogisticRegression": SGDClassifier(loss="log", random_state=random_state)
}

model_parameters = {
    "DecisionTreeClassifier": {
        'max_depth': [None, 5, 10],
        'min_samples_leaf': [1, 2, 3]
    },
    "KNeighborsClassifier": {
        'n_neighbors': [2, 3, 4],
        'weights': ['uniform', 'distance']
    },
    "SVC": {
        'C': [30, 40, 50],
        'kernel': ['poly', 'linear'],
        'degree': [2, 3, 4]
    },
    "LogisticRegression": {
        'alpha': [0.0001, 0.001, 0.01],
        'eta0': [0.1, 0.01, 0.05],
        'learning_rate': ['constant'],
        'penalty': ['l2', 'l1', 'none'],
        'max_iter': [200, 500, 1000]
    }
}

for name, parameters in model_parameters.items():
    model = models[name]
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(model, parameters, cv=cv, n_jobs=-1, verbose=False, scoring="accuracy")
    grid_search.fit(X_8, train_labels)
    
    print("Best parameters for {}: {}".format(name, grid_search.best_params_))
    print("Best accuracy score for {}: {:.2f}%".format(name, grid_search.best_score_ * 100))
```

    Best parameters for LogisticRegression: {'alpha': 0.0001, 'penalty': 'l1', 'eta0': 0.01, 'max_iter': 200, 'learning_rate': 'constant'}
    Best accuracy score for LogisticRegression: 88.08%
    Best parameters for DecisionTreeClassifier: {'max_depth': None, 'min_samples_leaf': 1}
    Best accuracy score for DecisionTreeClassifier: 77.49%
    Best parameters for KNeighborsClassifier: {'n_neighbors': 4, 'weights': 'distance'}
    Best accuracy score for KNeighborsClassifier: 89.04%
    Best parameters for SVC: {'C': 40, 'kernel': 'poly', 'degree': 3}
    Best accuracy score for SVC: 93.25%


#### Question 12:

There has been an increase in accuracy for DT, KNN and SVC. There hasn't been changes in the algorithm of SGD if we also consider the uncertainty in the scores. In general, tuning did not have significant impact in the models' performance. 


```python
print("Classifier\t\t\t\t After Tuning")
for name, parameters in model_parameters.items():
    model = models[name]
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(model, parameters, cv=cv, n_jobs=-1, verbose=False, scoring="accuracy")
    grid_search.fit(X_8, y)
    
    best_index = grid_search.best_index_
    std_dev = grid_search.cv_results_['std_test_score'][best_index]
    print("{}\t\t\t {:.2f} ± {:.2f}".format(name, grid_search.best_score_ * 100, std_dev * 100))
```

    Classifier				 After Tuning
    LogisticRegression			 88.08 ± 1.47
    DecisionTreeClassifier			 77.49 ± 1.80
    KNeighborsClassifier			 88.91 ± 1.57
    SVC			 92.93 ± 1.32


#### Question 13:

In general, the comparison between 8x8 and 28x28 data does not provide a significant advantage. SGD and the DT model has 8x8 dataset that outperforms the 28x28 dataset. In contrast, KNN and SVC has better accuracy in 28x28 dataset. 

The reason for this can be because more features can lead to more complexity which can lead to overfitting. Therefore, 8x8 dataset seems to have sufficient classification features to compare between models. 


```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

random_state = 42
n_splits = 10

models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=1),
    "SVC": SVC(random_state=random_state, gamma='scale'),
    "LogisticRegression": SGDClassifier(loss="log", random_state=random_state)
}

model_parameters = {
    "DecisionTreeClassifier": {
        'max_depth': [10, 30, 50],
        'min_samples_leaf': [1, 2, 4]
    },
    "KNeighborsClassifier": {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    },
    "SVC": {
        'C': [1, 10, 100],
        'kernel': ['poly', 'linear'],
        'degree': [2, 3, 4]
    },
    "LogisticRegression": {
        'alpha': [0.0001, 0.001, 0.01],
        'eta0': [0.1, 0.01, 0.001],
        'penalty': ['l2', 'l1', 'none'],
        'learning_rate' :["constant"],
        'max_iter': [1000, 5000, 10000]
    }
}

for name, parameters in model_parameters.items():
    model = models[name]
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    grid_search = GridSearchCV(model, parameters, cv=cv, n_jobs=-1, verbose=False, scoring="accuracy")
    grid_search.fit(X_28, train_labels)
    
    print("Best parameters for {}: {}".format(name, grid_search.best_params_))
    print("Best accuracy score for {}: {:.2f}%".format(name, grid_search.best_score_ * 100))
```

    Best parameters for LogisticRegression: {'alpha': 0.001, 'eta0': 0.01, 'learning_rate': 'constant', 'max_iter': 1000, 'penalty': 'l1'}
    Best accuracy score for LogisticRegression: 88.00%
    Best parameters for DecisionTreeClassifier: {'max_depth': 10, 'min_samples_leaf': 1}
    Best accuracy score for DecisionTreeClassifier: 74.77%
    Best parameters for KNeighborsClassifier: {'n_neighbors': 5, 'weights': 'distance'}
    Best accuracy score for KNeighborsClassifier: 89.09%
    Best parameters for SVC: {'degree': 3, 'kernel': 'poly', 'C': 100}
    Best accuracy score for SVC: 93.44%


#### Question 14:

Each algorithm performed differently with and without PCA. 

Both KNN and SVC has improve performance with PCA. These classifiers have improved due to the decreased noise allowing more informative features to be evaluated.

DT and SGD classifier has decreae with PCA. This might indicate that espeically SGD relies on the features that PCA has removed. 


```python
pca_8 = PCA(n_components = 15)
pca_8.fit(mnist_8x8_train_reshaped)
pca_8_transform = pca_8.transform(mnist_8x8_train_reshaped)


#models with hyperparameters that has been tuned 
dt_model = DecisionTreeClassifier(max_depth = None, min_samples_leaf=1, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=4, weights="distance")
svc_model = SVC(C=40, kernel="poly", degree=3, gamma='scale', random_state=42)
sgd_model = SGDClassifier(warm_start=True, loss='log', random_state=42, learning_rate='constant', 
                          eta0=0.01, penalty = 'l1', alpha=0.0001, max_iter=200)

#reasonable uncertainty of the estimate
def estimate_perf(clf, X_train_pca, y):
    scores = cross_val_score(clf, X_train_pca, y, cv = 10, scoring = 'accuracy')
    return np.mean(scores), np.std(scores)

perf = {}
for name, model in [("DT", dt_model),
                   ("KNN", knn_model),
                   ("SVC", svc_model),
                   ("SGD", sgd_model)]:
    mean, std = estimate_perf(model, pca_8_transform, y)
    perf[name] = (mean, std)

print("Classifier\t\tEstimated Generalization")
for name, (accuracy, std) in perf.items():
    print("{}\t\t\t {:.2f} ± {:.2f}".format(name, accuracy * 100, std * 100))
```

    Classifier		Estimated Generalization
    DT			 74.85 ± 1.92
    KNN			 91.97 ± 1.55
    SGD			 49.91 ± 4.89
    SVC			 92.75 ± 1.25


#### Question 15:


```python
X_test = mnist_28x28_test.reshape(mnist_28x28_test.shape[0], -1)

best_model = SVC(C=100, kernel="poly", degree=3, gamma='scale', random_state=42)
best_model.fit(X_28, train_labels)
prediction = best_model.predict(X_test)
estimate_accuracy = np.array([0.95])

result = np.append(estimate_accuracy, prediction)

pd.DataFrame(result).to_csv("accuracy_estimate_and_predictions_mnist.txt", index=False, header=False)
```
