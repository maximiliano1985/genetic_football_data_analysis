import sys
sys.path.append("./lib")

import itertools
import pandas as pd
import numpy as np
import matplotlib as plt


from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
#from pandas.tools.plotting import autocorrelation_plot

from core_functions import *
from predictor_engine import Predictor

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
matplotlib.use('WXAgg',warn=False, force=True)

##### General setup
N_MATCHES_TO_PREDICT = 30
out_classifier_filename = './models/football_classifier_serieA_DNN_extended'

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

print( "[PREPROC] Feature values:" )
print(X_all.head())

print( "[PREPROC] Labels values:" )
print(y_all.head())



# Separate training from test (i.e. to be predicted) data
ndata, _ = X_all.shape
X_train, y_train, X_valid, y_valid, X_test, y_test = split_datasets(X_all, y_all, valid_size = int(ndata*0.3), test_size = N_MATCHES_TO_PREDICT) 


# Instantiate a predictor 
print("[DNN] Created a predictor")
pred = Predictor({
            'out_classifier_filename': out_classifier_filename,
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
#regul_val = [1e-8,0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
regul_val = [1e-8];
pred.train( num_steps = 6001,
            regul_val = regul_val, # must be a list, even with one element
            verbose   = True,
            verbosity = 0.1 # statistics displayed every (num_steps*verbosity) iterations
        )

# Get results
y_train_reference = decode_probabilities(pred.train_labels_batch)
y_valid_reference = decode_probabilities(pred.valid_labels)
y_test_reference  = decode_probabilities(pred.test_labels)
y_train_predicted = decode_probabilities(pred.train_predictions)
y_valid_predicted = decode_probabilities(pred.valid_predictions)
y_test_predicted  = decode_probabilities(pred.test_predictions)

# Plot effect of regularization values
print("[DNN] Tested regularisations: ",regul_val)
print("[DNN] Validation accuracies: ",pred.valid_accuracies)



## Get predictions on the test set
teams_pred  = teams.iloc[-N_MATCHES_TO_PREDICT:, :]
teams_pred.index=teams_pred.index.tolist()
Pred_prob_A = []
Pred_prob_D = []
Pred_prob_H = []
for y in pred.test_predictions:
    Pred_prob_A.append(np.round(y[0], 3))
    Pred_prob_D.append(np.round(y[1], 3))
    Pred_prob_H.append(np.round(y[2], 3))

teams_pred['Predictions'] = y_test_predicted
teams_pred['Pred_prob_A'] = Pred_prob_A
teams_pred['Pred_prob_D'] = Pred_prob_D
teams_pred['Pred_prob_H'] = Pred_prob_H

teams_pred['Exact_res'] = y_test_reference

teams_pred['B365'], B365_OHE = one_hot_encoder_B365( data_pred )


print("__________________________")
print("[SIM] Accurancy BET365\t%.1f%%" % (calculate_accuracy(B365_OHE, pred.test_labels))) 
print("[SIM] Accurancy DNN\t\t%.1f%%" % (calculate_accuracy(np.array(pred.test_predictions), pred.test_labels))) 
print("__________________________")
print(teams_pred)


## Simulate profits with DNN and BET365 strategies
profits_B365 = []
profits_DNN = []

high_probab_thrs = 0.90
profits_DNN_highConf = []
for i, row in teams_pred.iterrows():    
    pred_B365 = row.B365
    pred_DNN  = row.Predictions
    res_exact = row.Exact_res
    
    row_data_pred = data_pred.loc[i]
    
    if pred_DNN == res_exact:
        if res_exact == 'H':
            profits_DNN.append(row_data_pred.B365H-1) # subtract 1 because I want the NET profit
        elif res_exact == 'D':
            profits_DNN.append(row_data_pred.B365D-1)
        elif res_exact == 'A':
            profits_DNN.append(row_data_pred.B365A-1)
    else:
        profits_DNN.append(-1) # I loose the euro I bet
    
    if max(row.Pred_prob_H, row.Pred_prob_D, row.Pred_prob_A) >= high_probab_thrs:# bet
        if pred_DNN == res_exact:
            if res_exact == 'H':
                profits_DNN_highConf.append(row_data_pred.B365H-1)
            elif res_exact == 'D':
                profits_DNN_highConf.append(row_data_pred.B365D-1)
            elif res_exact == 'A':
                profits_DNN_highConf.append(row_data_pred.B365A-1)
        else:
            profits_DNN_highConf.append(-1) # I loose the euro I bet
    else: # do not bet
        profits_DNN_highConf.append(0)

    
    
    if pred_B365 == res_exact:
        if res_exact == 'H':
            profits_B365.append(row_data_pred.B365H-1) 
        elif res_exact == 'D':
            profits_B365.append(row_data_pred.B365D-1)
        elif res_exact == 'A':
            profits_B365.append(row_data_pred.B365A-1) 
    else:
        profits_B365.append(-1) # I loose the euro I bet


teams_pred['profits_B365']          = profits_B365
teams_pred['profits_DNN']           = profits_DNN
teams_pred['profits_DNN_highConf']  = profits_DNN_highConf

print('[SIM] Estimated profits: BET365 %.2f €, DNN %.2f €'%(sum(teams_pred['profits_B365']), sum(teams_pred['profits_DNN'])))
print('[SIM] Estimated profits with high confidence: DNN_HC %.2f €'% (sum(teams_pred['profits_DNN_highConf'] ))) 

        
### PLOTS
# Visualising distribution of data
#plt.figure()#(figsize=(10, 6))
plt.subplot(2,3,1)
scm1 = scatter_matrix(X_all[['HTP', 'ATP', 'HTGD', 'ATGD', 'DiffFormPts', 'DiffLP', 'HomeScores', 'AwayScores']],
                     figsize=(10,10), diagonal='kde');
plt.title("Scatter matrix")
for ax in scm1.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 10, rotation = 90)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 10, rotation = 0)

# Dependecy of valid_calculate_accuracy on regul_val
if len(regul_val) > 1:
    plt.subplot(2,3,2)       
    plt.semilogx(regul_val, self.valid_accuracies)
    plt.grid(True)
    plt.title('Accuracy for different regul.values')


# Confusion matrix to check performance of classifier
cnf_matrix       = confusion_matrix( y_train_reference, y_train_predicted )
cnf_matrix_valid = confusion_matrix( y_valid_reference, y_valid_predicted )

np.set_printoptions(precision=2)
plt.subplot(2,3,3)
plot_confusion_matrix(cnf_matrix, classes=['Away', 'Draw', 'Home'], normalize=True)
plt.title('Confusion training set')

plt.subplot(2,3,6)
plot_confusion_matrix(cnf_matrix_valid, classes=['Away', 'Draw', 'Home'], normalize=True)
plt.title('Confusion validation set')
  
    
# Visualize performance predictions
plt.subplot(2,3,4)
teams_pred['profits_B365'].plot(color='red', kind='bar', label='B365', alpha=0.5)
teams_pred['profits_DNN'].plot(color='blue', kind='bar', label='DNN', alpha=0.5)
teams_pred['profits_DNN_highConf'].plot(color='green', kind='bar', label='DNN HC', alpha=0.5)
plt.axhline(0, color='k')
plt.title('Predicted profits')
plt.xlabel('Match ID')
plt.ylabel('€')
plt.legend()

ax = plt.subplot(2,3,5)
ax.plot( np.cumsum(teams_pred['profits_B365'].values.tolist()), 'r', label='Bet365')
ax.plot( np.cumsum(teams_pred['profits_DNN'].values.tolist()), 'b', label='DNN')
ax.plot( np.cumsum(teams_pred['profits_DNN_highConf'].values.tolist()), 'g', label='DNN HC')
plt.axhline(0, color='k')
plt.title('Predicted cumulated profits')
plt.xlabel('Match ID')
plt.ylabel('€')
plt.legend()

plt.tight_layout()
plt.show()
