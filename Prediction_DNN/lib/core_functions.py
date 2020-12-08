import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import itertools


def calculate_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def decode_probabilities(y):
    labels = []
    for pred in y:
        max_val = max(pred)
        if pred[0] == max_val:
            labels.append('A')
        elif pred[1] == max_val:
            labels.append('D')
        elif pred[2] == max_val:
            labels.append('H')
    return labels
    
    
def preprocess_features(X):
    X.HM1 = X.HM1.astype('str')
    X.HM2 = X.HM2.astype('str')
    X.HM3 = X.HM3.astype('str')
    X.AM1 = X.AM1.astype('str')
    X.AM2 = X.AM2.astype('str')
    X.AM3 = X.AM3.astype('str')
    
    ''' Preprocesses the football data and converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

def preprocess_labels(labels):
    labels =  pd.get_dummies(labels) # A [1, 0, 0] || D [0, 1, 0] || H [0, 0, 1]
    return labels
    
def scale_features(X):
    cols = X.keys().tolist()
    #cols = ['HTGD','ATGD','HTP','ATP','DiffLP', 'HomeAttack', 'HomeMedium', 'HomeDefense', 'HomeGK', 'AwayAttack',
    #       'AwayMedium', 'AwayDefense', 'HomeScores', 'AwayScores', 'HomeAwayDifference']
    for col in cols:
        X[col] = scale(X[col])/max(X[col])
        #X_all_raw[col] = X_all_raw[col]/max(X_all_raw[col])
    
    return X
    
def split_datasets(X, y, valid_size = 0.3, test_size = 0): # X: dataset, y: labels
    X_test = X.iloc[-test_size:, :]
    y_test = y[-test_size:]
    
    X_all = X.iloc[:-test_size, :]
    y_all = y[:-test_size]
    
    # Shuffle and split the dataset into training and testing set.
    X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                          test_size = valid_size,
                                                          random_state = 2)
                                                          
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues): 
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    title = 'Confusion matrix, without normalization'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    

    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def one_hot_encoder_B365(dataset):
    nmatches, _ = dataset.shape
    
    B365 = []
    B365_OHE = [] # one-hot encoded
    for i in range(nmatches):
        bm = dataset.iloc[i].B365H
        bd = dataset.iloc[i].B365D
        ba = dataset.iloc[i].B365A
        bmin = min(bm, bd, ba)
        if bm == bmin:
            B365.append('H')
            B365_OHE.append([0,0,1])
        elif bd == bmin:
            B365.append('D')
            B365_OHE.append([0,1,0])

        elif ba == bmin:
            B365.append('A')
            B365_OHE.append([1,0,0])
        else:
            B365.append('Error')
            B365_OHE.append([0,0,0])
    return B365, np.array(B365_OHE)