import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('dermatology.data', delimiter=',', header =None)

column_names = ['erythema', 'scaling', 'definite borders', 'itching', 'koebner phenomenon', 'polygonal papules', 'follicular papules', 'oral mucosal involvement',
                'knee and elbow involvement', 'scalp involvement', 'family history', 'melanin incontinence', 'eosinophils in the infiltrate', 'PNL infiltrate',
                'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing of the rete ridges', 
                'elongation of the rete ridges', 'thinning of the suprapapillary epidermis', 'spongiform pustule', 'munro microabcess', 'focal hypergranulosis',
                'disappearance of the granular layer', 'vacuolisation and damage of basal layer', 'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug',
                'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate', 'band-like infiltrate', 'Age', 'Class']

dataset.columns = column_names

# Replacing ? in Age column with median age
# 8 out of 366 rows
dataset['Age'] = dataset['Age'].replace('?', np.nan)
median_age = dataset['Age'].median()
dataset['Age'].fillna(median_age, inplace=True)


# Replacing 1-5 with 0 and 6 with 1 for Class
# Only looking for benign vs malignant classes
dataset['Class'] = dataset['Class'].replace([1, 2, 3, 4, 5], 0)
dataset['Class'] = dataset['Class'].replace(6, 1)


# Visualization of variable distribution
for i in range(34):
    sns.histplot(data=dataset.iloc[:, i])
    plt.show()
    sns.boxplot(data=dataset.iloc[:, i])
    plt.title(column_names[i])
    plt.show()


# Assigning input features and output label
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.decomposition import PCA
# good practice is to start with n components = 2 and if results are bad, increase the amount of components
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


# Training the Logistic Regresssion Classifier model on the Training set
from sklearn.linear_model import LogisticRegression
classifierLogRes=LogisticRegression(random_state= 0)
classifierLogRes.fit(X_train, y_train)

# Training the Kernel SVM Classifier model on the Training set
from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'rbf', random_state = 0, probability=True)
classifierSVC.fit(X_train, y_train)

# Training the Naive Bayes Classifier model on the Training set
from sklearn.naive_bayes import GaussianNB
classifierNB=GaussianNB()
classifierNB.fit(X_train, y_train)

# Training the Random Forest Classifier model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRFC.fit(X_train, y_train)


#### Making and training an Artificial Neural Network
import tensorflow as tf
# Initialize ANN
ann=tf.keras.models.Sequential()
# Add first/input layer of neural network equal to the number of input features
ann.add(tf.keras.layers.Dense(units=33,activation='relu'))
# Add second/hidden layer of neural network
ann.add(tf.keras.layers.Dense(units=7,activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) 

# Compiling the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training the ANN on the training set
ann.fit(X_train,y_train,epochs=100,batch_size=32)


# Making the Confusion Matrix for each classifier model
from sklearn.metrics import confusion_matrix, accuracy_score

y_predLogRes = classifierLogRes.predict(X_test)
cmLogRes = confusion_matrix(y_test, y_predLogRes)

y_predSVC = classifierSVC.predict(X_test)
cmSVC = confusion_matrix(y_test, y_predSVC)

y_predNB = classifierNB.predict(X_test)
cmNB = confusion_matrix(y_test, y_predNB)

y_predRFC = classifierRFC.predict(X_test)
cmRFC = confusion_matrix(y_test, y_predRFC)

y_predANN = ann.predict(X_test)
y_predANN = (y_predANN > 0.5)
cmANN = confusion_matrix(y_test, y_predANN)


classifierCM = [cmLogRes, cmSVC, cmNB, cmRFC, cmANN] 
classifierNames = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'Random Forest', 'Artificial Neural Network']

# Assuming cm is the confusion matrix you want to plot
for cf, cfName in zip(classifierCM, classifierNames):
    sns.set(font_scale=1.4) # Adjust the font scale for better readability
    sns.heatmap(cf, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
    # Set axis labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{cfName}')
    plt.show()



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifierLogRes,X=X_train,y=y_train,cv=10)
print("Logistic Regression:\n\tAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("\tStandard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=classifierSVC,X=X_train,y=y_train,cv=10)
print("Support Vector Machines:\n\tAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("\tStandard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=classifierNB,X=X_train,y=y_train,cv=10)
print("Gaussian Naive Bayes:\n\tAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("\tStandard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=classifierRFC,X=X_train,y=y_train,cv=10)
print("Random Forest:\n\tAccuracy: {:.2f} %".format(accuracies.mean()*100))
print("\tStandard Deviation: {:.2f} %".format(accuracies.std()*100))


classifierObjects = [classifierLogRes, classifierSVC, classifierNB, classifierRFC, ann]

# Visualizing test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
for cf, cfName in zip(classifierObjects, classifierNames):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, cf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(f'{cfName} (Training set)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()






# Plots for ROC, PRC, AUC
from sklearn.metrics import roc_curve, auc

# Example data
#y_true = np.array([0, 0, 1, 1])
for cf, cfName in zip(classifierObjects, classifierNames):
    if cfName == "Artificial Neural Network":
        y_score = cf.predict(X_test)
    else:
        y_score = cf.predict_proba(X_test)[:, 1]
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{cfName}: receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()