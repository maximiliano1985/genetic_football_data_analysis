import sys
sys.path.append("./lib")
import itertools
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
#from pandas.tools.plotting import autocorrelation_plot
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

from core_functions import *
from predictor_engine import Predictor


import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
matplotlib.use('WXAgg',warn=False, force=True)

##### General setup
N_MATCHES_TO_PREDICT = 30

out_classifier_filename = './tmp/football_classifier_serieA_DNN_extended'
out_classifiers_filenames = []
for i in range(10):
    out_classifiers_filenames.append(out_classifier_filename+str(i))

##### Load the data
in_data_filename = '../Datasets/serie_A/final_dataset_serieA_extended.csv'
data = pd.read_csv(in_data_filename)
print("[PREPROC] Loaded data from %s"%(in_data_filename))

##### Define the features
#cols = ['FTR', 'HTP', 'ATP', 'HTGD', 'ATGD', 'DiffFormPts', 'DiffLP', 'HomeScore_PrevW1', 'AwayScore_PrevW1',
#        'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3']
#data = data[cols]

# Remove first 3 matchweeks
data = data[data.MW > 3]

data = data[:-10] # no statistics are known for last 10 data (1 week of game),
                  # therefore these are neglected since cannot be used to estimate
                  # the return of the bets.

data_pred = data[-N_MATCHES_TO_PREDICT:] # data to be used when assessing
                                         # the economical performance of the estimations

teams = pd.concat([data['HomeTeam'], data['AwayTeam']], axis=1, keys=['HomeTeam', 'AwayTeam'])


data.drop(['Unnamed: 0','HomeTeam', 'AwayTeam','Date', 'MW', 'HTFormPtsStr', 'ATFormPtsStr', 'FTHG', 'FTAG',
           'HTGS', 'ATGS', 'HTGC', 'ATGC','HomeTeamLP', 'AwayTeamLP','DiffPts','HTFormPts','ATFormPts',
           'HM4','AM4','HTLossStreak5','ATLossStreak5','HTWinStreak5','ATWinStreak5',
           'HTWinStreak3','HTLossStreak3','ATWinStreak3','ATLossStreak3', 'ENTROPY_B365', 'ENTROPY_BW',
           'ENTROPY_IW', 'ENTROPY_LB', 'ENTROPY_WH', 'ENTROPY_H', 'ENTROPY_D', 'ENTROPY_A',
           'B365H', 'BWH', 'IWH', 'LBH', 'WHH', 'B365D', 'BWD', 'IWD', 'LBD', 'WHD', 'B365A', 'BWA', 'IWA',
           'LBA', 'WHA','PSA', 'PSCA', 'PSCD', 'PSCH', 'PSD', 'PSH', 'SBA', 'SBD', 'SBH', 'SJA', 'SJD',
           'SJH', 'VCA', 'VCD', 'VCH', 'GBA', 'GBD', 'GBH', 'HC', 'HF', 'AC', 'AF', 'BbAv<2.5', 'BbAv>2.5',
           'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'AY',
           'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU', 'Div', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'AR', 'AS', 'AST',
           'BSA', 'BSD', 'BSH', 'Bb1X2', 'BbAH', 'BbAHh', 'HTR', 'HY',
           'HomeAttack', 'HomeMedium', 'HomeDefense', 'HomeGK', 'HomeAwayDifference',
           'AwayAttack', 'AwayMedium', 'AwayDefense', 'AwayGK','HomeScore_PrevW1', 'AwayScore_PrevW1'],1, inplace=True)


# Separate into feature set and target variable
X_all = data.drop(['FTR'],1) # dataset
y_all = data['FTR'] # labels

# Preview data.
print('[PREPROC] Features:')
print(X_all.keys())

##### Features preprocessing
X_all = preprocess_features(X_all)
y_all = preprocess_labels(y_all)
print( "[PREPROC] Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)) )

# Standardise the data.
X_all = scale_features(X_all)
print( "[PREPROC] Feature stats:" )
print(X_all.describe())

#print( "[PREPROC] Feature values:" )
#print(X_all.head())
#
#print( "[PREPROC] Labels values:" )
#print(y_all.head())


##########
for classifier_filename in out_classifiers_filenames:
    print("[DNN] ##### CLASSIFIER:", classifier_filename, "#####")
    
    # Separate training from test (i.e. to be predicted) data
    ndata, _ = X_all.shape
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_datasets(X_all, y_all, valid_size = int(ndata*0.3), test_size = N_MATCHES_TO_PREDICT) 
    
    
    # Instantiate a predictor 
    print("[DNN] Created a predictor")
    pred = Predictor({
                'out_classifier_filename': classifier_filename,
                'batch_size' : 200,
                'X_train' : X_train,
                'y_train' : y_train,
                'X_valid' : X_valid,
                'y_valid' : y_valid,
                'X_test'  : X_test,
                'y_test'  : y_test,
            })
            
    # Create a model
    hidden_layers_3 = [
       {'nnodes':20, 'keep_prob':0.5}, # 1st hidden layer
       {'nnodes':10, 'keep_prob':0.5}, # 2nd hidden layer
       {'nnodes':5 , 'keep_prob':0.5}, # 3rd hidden layer
    ]
    hidden_layers_4 = [
       {'nnodes':300, 'keep_prob':0.5}, # 1st hidden layer
       {'nnodes':100, 'keep_prob':0.5}, # 2nd hidden layer
       {'nnodes':50 , 'keep_prob':0.5}, # 3rd hidden layer
       {'nnodes':10 , 'keep_prob':0.5}, # 3rd hidden layer
    ]
    pred.build_model(hidden_layers_3)
    
    # Train the network
    regul_val = [1e-8,0.0001, 0.0005, 0.001, 0.005]
    #regul_val = [1e-8];
    pred.train( num_steps = 30001,
                regul_val = regul_val, # must be a list, even with one element
                verbose   = True,
                verbosity = 0.1 # statistics displayed every (num_steps*verbosity) iterations
            )
    # Plot effect of regularization values
    print("[DNN] Tested regularisations: ",regul_val)
    print("[DNN] Validation accuracies: ",pred.valid_accuracies)


