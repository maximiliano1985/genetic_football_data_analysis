############### Import the necessary libraries.
import sys
sys.path.append("./lib")
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
from time import time
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'


from core_functions import *

DECODER = {0:'A', 1:'D', 2:'H'}

############### General setup
N_MATCHES_TO_PREDICT = 10

in_classifier_filename = './tmp/football_classifier_serieA_DNN_extended'
in_data_filename = '../Datasets/serie_A/final_dataset_serieA_extended.csv'

out_prediction_filename = './predictions/preds.csv'

############### preprocess
in_classifiers_filenames = []
for i in range(10):
    in_classifiers_filenames.append(in_classifier_filename+str(i))

############### Load the data
data = pd.read_csv(in_data_filename)
print("[PREPROC] Loaded data from %s"%(in_data_filename))


############### Assign penalty or boosting to the teams scores
# This is done to consider the fact that the players might be tired or have a very high motivation
#indx = data.index[(data.HomeTeam == 'Chievo') & (data.AwayTeam == 'Napoli')].tolist()[-1] # champions
#data.AwayAttack[indx]   = data.AwayAttack[indx]*0.5
#data.AwayMedium[indx]  = data.AwayMedium[indx]*0.5
#data.AwayDefense[indx] = data.AwayDefense[indx]*0.5
#data.AwayGK[indx]      = data.AwayGK[indx]


############### Preparing the Data
# Remove first 3 matchweeks
data = data[data.MW > 3]

teams = pd.concat([data['Date'], data['HomeTeam'], data['AwayTeam']], axis=1, keys=['Date', 'HomeTeam', 'AwayTeam'])

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

############### Features preprocessing
X_all = preprocess_features(X_all)
y_all = preprocess_labels(y_all)
print( "[PREPROC] Processed feature columns ({} total features:\n{}".format(len(X_all.columns), list(X_all.columns)) )

# Standardise the data.
X_all = scale_features(X_all)
print( "[PREPROC] Feature stats:" )
print(X_all.describe())


# Separate training from test (i.e. to be predicted) data
X_test      = X_all.iloc[-N_MATCHES_TO_PREDICT:, :]
y_test      = y_all[-N_MATCHES_TO_PREDICT:]
ntest, _    = X_test.shape

print("Matches to predict: last %d"%( ntest))
test_dataset  = X_test.values.tolist()
test_labels   = y_test.values.tolist()


############### Load the Deep Neural Network
test_predictions = []
for classifier_filename in in_classifiers_filenames:
    print("[DNN] Predicting with", classifier_filename)
    
    test_pred = []
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(classifier_filename+'.meta')
        saver.restore(session, classifier_filename)

        tf.get_default_graph().as_graph_def()
    
        tf_test_dataset = session.graph.get_tensor_by_name("predict_dataset:0")
        test_prediction = session.graph.get_tensor_by_name("predict:0")

        for y in test_dataset:
            feed_dict = {tf_test_dataset : [y]}
            test_pred.append( list( session.run( [test_prediction], feed_dict=feed_dict)[0][0] ) )
    
    test_predictions.append(test_pred)
print(test_predictions)


############### Average the results
teams_pred  = teams.iloc[-N_MATCHES_TO_PREDICT:, :]
teams_pred.index=teams_pred.index.tolist()

pred_prob_A = []; pred_prob_A_std = []
pred_prob_D = []; pred_prob_D_std = []
pred_prob_H = []; pred_prob_H_std = []
predictions = []
df_plots    = []
for match_idx in range(len(test_dataset)):
    num_nets = len(test_predictions)
    prob_A = []
    prob_D = []
    prob_H = []
    for pred in test_predictions:
        prob_A.append( pred[ match_idx ][0] )
        prob_D.append( pred[ match_idx ][1] )
        prob_H.append( pred[ match_idx ][2] )
    
    df_plots.append( pd.DataFrame({'H':prob_H, 'D':prob_D, 'A':prob_A}) )
        
    avg_prob_A = np.average(prob_A)
    avg_prob_D = np.average(prob_D)
    avg_prob_H = np.average(prob_H)
    pred_prob_A.append( np.round( avg_prob_A, 3) )
    pred_prob_D.append( np.round( avg_prob_D, 3) )
    pred_prob_H.append( np.round( avg_prob_H, 3) )
    pred_prob_A_std.append( np.round( np.std(prob_A), 3) )
    pred_prob_D_std.append( np.round( np.std(prob_D), 3) )
    pred_prob_H_std.append( np.round( np.std(prob_H), 3) )
    
    predictions.append( DECODER[ np.argmax([avg_prob_A, avg_prob_D, avg_prob_H]) ] )

teams_pred['Exact_res'] = decode_probabilities(test_labels)
teams_pred['Predictions'] = predictions
teams_pred['Pred_prob_A'] = pred_prob_A
teams_pred['Pred_prob_D'] = pred_prob_D
teams_pred['Pred_prob_H'] = pred_prob_H
teams_pred['Std_A'] = pred_prob_A_std
teams_pred['Std_D'] = pred_prob_D_std
teams_pred['Std_H'] = pred_prob_H_std


############### Output the results
res = teams_pred[['Date', 'HomeTeam','AwayTeam', 'Exact_res', 'Predictions', 'Pred_prob_H', 'Pred_prob_D', 'Pred_prob_A', 'Std_H', 'Std_D', 'Std_A']]
res.to_csv(out_prediction_filename)
print('=================================')
print(res)

plt.figure(figsize=(8, 18))
for indx, df in enumerate( df_plots ):
    plt.subplot(5,2,indx+1)
    df.boxplot(positions=[2, 1, 0])
    title = teams_pred.iloc[indx]['HomeTeam'] + ' vs ' + teams_pred.iloc[indx]['AwayTeam']
    plt.title( title )
    plt.xticks([0,1,2], ['H', 'D', 'A'])
    plt.ylim(-0.3, 1.3)

plt.savefig('./predictions/preds.png')
#plt.show()


