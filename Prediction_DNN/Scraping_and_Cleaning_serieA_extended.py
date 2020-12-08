# Import all the necessary libraries
import sys
sys.path.append("./lib")

import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
import copy
from data_proc import *
pd.options.mode.chained_assignment = None  # default='warn'

# Read data from the CSV into a dataframe

loc = "../Datasets/serie_A/"

in_filestandings = loc + "ISAStandings.csv"
Standings = pd.read_csv(in_filestandings)  
known_teams = Standings.Team

out_filename = loc + "final_dataset_serieA_extended.csv"


in_scorespath = '../Datasets/serie_A/whoscored_data/'



print("Load data")
raw_data_10 = pd.read_csv(loc + '2009-10.csv')
raw_data_11 = pd.read_csv(loc + '2010-11.csv')
raw_data_12 = pd.read_csv(loc + '2011-12.csv')
raw_data_13 = pd.read_csv(loc + '2012-13.csv')
raw_data_14 = pd.read_csv(loc + '2013-14.csv')
raw_data_15 = pd.read_csv(loc + '2014-15.csv')
raw_data_16 = pd.read_csv(loc + '2015-16.csv')
raw_data_17 = pd.read_csv(loc + '2016-17.csv')
raw_data_18 = pd.read_csv(loc + '2017-18.csv')

raw_scores_10 = pd.read_csv(in_scorespath + 'scores_2009-2010.csv')
raw_scores_11 = pd.read_csv(in_scorespath + 'scores_2010-2011.csv')
raw_scores_12 = pd.read_csv(in_scorespath + 'scores_2011-2012.csv')
raw_scores_13 = pd.read_csv(in_scorespath + 'scores_2012-2013.csv')
raw_scores_14 = pd.read_csv(in_scorespath + 'scores_2013-2014.csv')
raw_scores_15 = pd.read_csv(in_scorespath + 'scores_2014-2015.csv')
raw_scores_16 = pd.read_csv(in_scorespath + 'scores_2015-2016.csv')
raw_scores_17 = pd.read_csv(in_scorespath + 'scores_2016-2017.csv')
raw_scores_18 = pd.read_csv(in_scorespath + 'scores_2017-2018.csv')

print("Parse dates")
raw_data_10.Date = raw_data_10.Date.apply(parse_date)
raw_data_11.Date = raw_data_11.Date.apply(parse_date)
raw_data_12.Date = raw_data_12.Date.apply(parse_date)
raw_data_13.Date = raw_data_13.Date.apply(parse_date)
raw_data_14.Date = raw_data_14.Date.apply(parse_date)
raw_data_15.Date = raw_data_15.Date.apply(parse_date)
raw_data_16.Date = raw_data_16.Date.apply(parse_date)
raw_data_17.Date = raw_data_17.Date.apply(parse_date)
raw_data_18.Date = raw_data_18.Date.apply(parse_date)

playing_statistics_10 = raw_data_10
playing_statistics_11 = raw_data_11
playing_statistics_12 = raw_data_12
playing_statistics_13 = raw_data_13
playing_statistics_14 = raw_data_14
playing_statistics_15 = raw_data_15
playing_statistics_16 = raw_data_16
playing_statistics_17 = raw_data_17
playing_statistics_18 = raw_data_18


#** GOALS SCORED AND CONCEDED AT THE END OF MATCHWEEK, ARRANGED BY TEAMS AND MATCHWEEK  **
print("Getting scored and conceded goals")

# Apply to each dataset
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)
playing_statistics_14 = get_gss(playing_statistics_14)
playing_statistics_15 = get_gss(playing_statistics_15)
playing_statistics_16 = get_gss(playing_statistics_16)
playing_statistics_17 = get_gss(playing_statistics_17)
playing_statistics_18 = get_gss(playing_statistics_18)

print("Calculating teams points")
# Apply to each dataset
playing_statistics_10 = get_agg_points(playing_statistics_10)
playing_statistics_11 = get_agg_points(playing_statistics_11)
playing_statistics_12 = get_agg_points(playing_statistics_12)
playing_statistics_13 = get_agg_points(playing_statistics_13)
playing_statistics_14 = get_agg_points(playing_statistics_14)
playing_statistics_15 = get_agg_points(playing_statistics_15)
playing_statistics_16 = get_agg_points(playing_statistics_16)
playing_statistics_17 = get_agg_points(playing_statistics_17)
playing_statistics_18 = get_agg_points(playing_statistics_18)

print("Aggregating teams' scores on playing areas")
playing_statistics_10 = compute_team_scores(playing_statistics_10, raw_scores_10)     
playing_statistics_11 = compute_team_scores(playing_statistics_11, raw_scores_11)     
playing_statistics_12 = compute_team_scores(playing_statistics_12, raw_scores_12)     
playing_statistics_13 = compute_team_scores(playing_statistics_13, raw_scores_13)     
playing_statistics_14 = compute_team_scores(playing_statistics_14, raw_scores_14)     
playing_statistics_15 = compute_team_scores(playing_statistics_15, raw_scores_15)     
playing_statistics_16 = compute_team_scores(playing_statistics_16, raw_scores_16)     
playing_statistics_17 = compute_team_scores(playing_statistics_17, raw_scores_17)     
playing_statistics_18 = compute_team_scores(playing_statistics_18, raw_scores_18)  

print("Reordering teams' scores on playing areas")
playing_statistics_10 = get_scores_previous_weeks(playing_statistics_10, 1)     
playing_statistics_11 = get_scores_previous_weeks(playing_statistics_11, 1)     
playing_statistics_12 = get_scores_previous_weeks(playing_statistics_12, 1)     
playing_statistics_13 = get_scores_previous_weeks(playing_statistics_13, 1)     
playing_statistics_14 = get_scores_previous_weeks(playing_statistics_14, 1)     
playing_statistics_15 = get_scores_previous_weeks(playing_statistics_15, 1)     
playing_statistics_16 = get_scores_previous_weeks(playing_statistics_16, 1)     
playing_statistics_17 = get_scores_previous_weeks(playing_statistics_17, 1)     
playing_statistics_18 = get_scores_previous_weeks(playing_statistics_18, 1)  

print("Calculating teams statistics")
# Make changes to df
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_14 = add_form_df(playing_statistics_14)
playing_statistics_15 = add_form_df(playing_statistics_15)
playing_statistics_16 = add_form_df(playing_statistics_16)
playing_statistics_17 = add_form_df(playing_statistics_17)
playing_statistics_18 = add_form_df(playing_statistics_18)

   
#print(playing_statistics_18.iloc[-20:,:])

print("Predicting scores for teams' sections...")
playing_statistics_18 = fill_teams_pts(playing_statistics_18)

# Rearranging columns
#cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'HM4', 'AM1', 'AM2', 'AM3', 'AM4', 'B365H', 'BWH', 'IWH', 'LBH', 'WHH', 'B365D','BWD','IWD','LBD', 'WHD', 'B365A','BWA','IWA','LBA',  'WHA', 'HomeAttack', 'HomeMedium', 'HomeDefense', 'HomeGK', 'AwayAttack', 'AwayMedium', 'AwayDefense', 'AwayGK']

#playing_statistics_10 = playing_statistics_10[cols]
#playing_statistics_11 = playing_statistics_11[cols]
#playing_statistics_12 = playing_statistics_12[cols]
#playing_statistics_13 = playing_statistics_13[cols]
#playing_statistics_14 = playing_statistics_14[cols]
#playing_statistics_15 = playing_statistics_15[cols]
#playing_statistics_16 = playing_statistics_16[cols]
#playing_statistics_17 = playing_statistics_17[cols]
#playing_statistics_18 = playing_statistics_18[cols]

#Get Last Year's Position as also an independent variable:
print('Including standings information')  
Standings.set_index(['Team'], inplace=True)
Standings = Standings.fillna(22) # the teams that did not play the championship get a high (bad) ranking

playing_statistics_10 = get_last(playing_statistics_10, Standings, 10)
playing_statistics_11 = get_last(playing_statistics_11, Standings, 11)
playing_statistics_12 = get_last(playing_statistics_12, Standings, 12)
playing_statistics_13 = get_last(playing_statistics_13, Standings, 13)
playing_statistics_14 = get_last(playing_statistics_14, Standings, 14)
playing_statistics_15 = get_last(playing_statistics_15, Standings, 15)
playing_statistics_16 = get_last(playing_statistics_16, Standings, 16)
playing_statistics_17 = get_last(playing_statistics_17, Standings, 17)
playing_statistics_18 = get_last(playing_statistics_18, Standings, 18)

playing_statistics_10 = get_mw(playing_statistics_10)
playing_statistics_11 = get_mw(playing_statistics_11)
playing_statistics_12 = get_mw(playing_statistics_12)
playing_statistics_13 = get_mw(playing_statistics_13)
playing_statistics_14 = get_mw(playing_statistics_14)
playing_statistics_15 = get_mw(playing_statistics_15)
playing_statistics_16 = get_mw(playing_statistics_16)
playing_statistics_17 = get_mw(playing_statistics_17)
playing_statistics_18 = get_mw(playing_statistics_18)

        
print("Get bookmakers entropies")
playing_statistics_10 = compute_entropy(playing_statistics_10)
playing_statistics_11 = compute_entropy(playing_statistics_11)
playing_statistics_12 = compute_entropy(playing_statistics_12)
playing_statistics_13 = compute_entropy(playing_statistics_13)
playing_statistics_14 = compute_entropy(playing_statistics_14)
playing_statistics_15 = compute_entropy(playing_statistics_15)
playing_statistics_16 = compute_entropy(playing_statistics_16)
playing_statistics_17 = compute_entropy(playing_statistics_17)
playing_statistics_18 = compute_entropy(playing_statistics_18)




#FINAL DATAFRAME
playing_stat = pd.concat([
    playing_statistics_10,
    playing_statistics_11,
    playing_statistics_12,
    playing_statistics_13,
    playing_statistics_14,
    playing_statistics_15,
    playing_statistics_16,
    playing_statistics_17,
    playing_statistics_18], ignore_index=True)





#playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
#playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']
playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4']

# convert the string containing the results from the last 5 weeks into a total score
playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)



print("Get additional features")
playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)


print("playing_stat.keys()\n", playing_stat.keys())

# Get Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Diff in points
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

# Diff in last year positions
playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']

#print(playing_stat.iloc[-20:,:])

# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# to simplify the problem try only to guess if the home team is going to win
#playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)
playing_stat.to_csv(out_filename)

print('Saved dataset in: ', out_filename)
print("Done!")
