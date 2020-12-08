from datetime import datetime as dt
import itertools
import copy
from scipy.stats import entropy
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


# Parse data as time 
def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%y').date()
    

def parse_date_other(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%Y').date()
        
        
#** GOALS SCORED AND CONCEDED AT THE END OF MATCHWEEK, ARRANGED BY TEAMS AND MATCHWEEK  **
     
# Gets the goals scored agg arranged by teams and matchweek
def get_goals_scored(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in playing_stat.groupby('AwayTeam').mean().T.columns:
        teams[i] = []   
         
    # the value corresponding to keys is a list containing the match location.
    # for every team get the number of goals made on each match and list them into a list
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG'] # FTHG = Full Time Home Team Goals
        ATGS = playing_stat.iloc[i]['FTAG'] # FTAG = Full Time Away Team Goals
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    # consider also the amount of matches played by all the teams
    value = list(teams.values())[0]
    n_matches = len(value) # 39 = list(teams.values())[0]
    for k, v in teams.items():
        if len(v) != n_matches:
            n_matches = min(len(v), n_matches)                     
    for k, v in teams.items():
        if len(v) > n_matches:
            teams[k] = v[:n_matches]
            
    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,n_matches+1)]).T
    GoalsScored[0] = 0
    #print("NEW GoalsScored ", GoalsScored)
    
    # Cumulate to get uptil that point
    for i in range(2,n_matches+1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]        
    return GoalsScored
    
# Gets the goals conceded agg arranged by teams and matchweek
def get_goals_conceded(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in playing_stat.groupby('AwayTeam').mean().T.columns:
        teams[i] = []   
        
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    
    # consider also the amount of matches played by all the teams
    value = list(teams.values())[0]
    n_matches = len(value) # 39 = list(teams.values())[0]
    for k, v in teams.items():
        if len(v) != n_matches:
            n_matches = min(len(v), n_matches)                     
    for k, v in teams.items():
        if len(v) > n_matches:
            teams[k] = v[:n_matches]
    
    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,n_matches+1)]).T
    GoalsConceded[0] = 0
    # Aggregate to get uptil that point
    for i in range(2,n_matches+1):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    
    return GoalsConceded



def get_gss(playing_stat):
    # playing_stat è il dataframe del foglio excel
    # GC e GS riportano il numero cumulato di goal fatti ogni settimana
    GC = get_goals_conceded(playing_stat)#, copy.deepcopy(teams))
    GS = get_goals_scored(playing_stat)#, copy.deepcopy(teams))
    
    j = 0
    HTGS = [] # home team goals scored
    ATGS = [] # away team goals scored
    HTGC = [] # home team goals conceded
    ATGC = [] # away team goals conceded
    total_matches, nmatches = playing_stat.shape
    nteams, nweeksp1 = GS.shape
    nmatches_per_week = nteams/2
    
    for i in range( total_matches ): 
        row = playing_stat.iloc[i]
        ht = row.HomeTeam
        at = row.AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        # every week there are 10 matches
        if ((i + 1)% nmatches_per_week) == 0:
            j = j + 1
        #print("i ", i, ", j ", j, ", playing_stat.iloc[i] ", playing_stat.iloc[i], "\nHTGS ", HTGS)
    

    # add to playing_stats these four columns, so for every match are available the 
    # cumulative values of goals scored or conceded for the teams involved in the match
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    return playing_stat
    
    
#GET RESPECTIVE POINTS:
# one-hot encoding of the results. These will be used as performance scores
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
    
# get the cumulatives of the performance scores for each team
def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    nteams, nmatches = matchres_points.shape
    for i in range(2,nmatches+1): # starts from 2 because: col_0 is the team, col_1 is the first match
        matchres_points[i] = matchres_points[i] + matchres_points[i-1] # cumulate the encodings (i.e. the performance scores)
    
    # add a column of zeros for the week number '0'
    matchres_points.insert(column = 0, loc = 0, value = [0*i for i in range(nteams)])
    return matchres_points


def get_matchres(playing_stat):
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in playing_stat.groupby('AwayTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            
    # consider also the amount of matches played by all the teams
    value = list(teams.values())[0]
    n_matches = len(value) # 39 = list(teams.values())[0]
    for k, v in teams.items():
        if len(v) != n_matches:
            n_matches = min(len(v), n_matches)                     
    for k, v in teams.items():
        if len(v) > n_matches:
            teams[k] = v[:n_matches]
    
    # dataframe containing for every team, the results (win/draw/loose) of the matches
    # played every wheek
    return pd.DataFrame(data=teams, index = [i for i in range(1,n_matches+1)]).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = [] # home team points (points are the performance scores)
    ATP = [] # away team points
    j = 0

    total_matches, nmatches = playing_stat.shape
    nteams, nweeks = cum_pts.shape
    nmatches_per_week = nteams/2

    
    for i in range( total_matches ):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% nmatches_per_week) == 0:
            j = j + 1
    
    # add the points for each team   
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat
    
    
#GET TEAM FORM:
# for every team and every week create a string contatenating the last 'num' matchres
def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    nteams, nmatches = form_final.shape
    for i in range(num,nmatches+1):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat,num):
    
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10). Nota that there are 10 matches per week
    a = ['M' for i in range(num * 10)]

    j = num
    total_matches, nmatches = playing_stat.shape
    for i in range((num*10),total_matches):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j] # get past n results
        h.append(past[num-1])  # 0 index is most recent
        
        past = form.loc[at][j] # get past n results.
        a.append(past[num-1])  # 0 index is most recent
        
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a    
    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1) # add a column with the results from the previous week
    playing_statistics = add_form(playing_statistics,2) # add a column with the results from the previous 2 weeks
    playing_statistics = add_form(playing_statistics,3) # add a column with the results from the previous 3 weeks
    playing_statistics = add_form(playing_statistics,4) # add a column with the results from the previous 4 weeks
    #playing_statistics = add_form(playing_statistics,5) # add a column with the results from the previous 5 weeks
    return playing_statistics    
    
    
# get points at the end of the season 'year'
def get_last(playing_stat, Standings, year):
    HomeTeamLP = []
    AwayTeamLP = []
    total_matches, nmatches = playing_stat.shape

    #print(Standings) ###
    for i in range(total_matches):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HomeTeamLP.append(Standings.loc[ht][year])
        AwayTeamLP.append(Standings.loc[at][year])
        
    playing_stat['HomeTeamLP'] = HomeTeamLP
    playing_stat['AwayTeamLP'] = AwayTeamLP
    return playing_stat    


# add the matchweek to playing_stat
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    total_matches, nmatches = playing_stat.shape
    for i in range(total_matches):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat


def match_entropy(row, bookmaker):
    odds = [row[bookmaker+'H'],row[bookmaker+'D'],row[bookmaker+'A']]
    #change odds to probability
    probs = [1/o for o in odds]
    #normalize to sum to 1
    norm = sum(probs)
    probs = [p/norm for p in probs]
    return entropy(probs)

def match_entropies(row):
    odds_h = row[['B365H','BWH','IWH','LBH', 'WHH']].values
    odds_d = row[['B365D','BWD','IWD','LBD', 'WHD']].values
    odds_a = row[['B365A','BWA','IWA','LBA', 'WHA']].values
    
    #change odds to probability
    probs_h = [1/o for o in odds_h]
    probs_d = [1/o for o in odds_d]
    probs_a = [1/o for o in odds_a]


    #normalize to sum to 1
    norm_h = sum(probs_h)
    norm_d = sum(probs_d)
    norm_a = sum(probs_a)
    probs_h = [p/norm_h for p in probs_h]
    probs_d = [p/norm_d for p in probs_d]
    probs_a = [p/norm_a for p in probs_a]
    return entropy(probs_h), entropy(probs_d), entropy(probs_a)
    

def compute_entropy(playing_stat):
    ENTROPY_B365 = []
    ENTROPY_BW   = []
    ENTROPY_IW   = []
    ENTROPY_LB   = []
    ENTROPY_WH   = []
    ENTROPY_H = []
    ENTROPY_D = []
    ENTROPY_A = []
    
    total_matches, nmatches = playing_stat.shape
    
    for i in range( total_matches ): 
        row = playing_stat.iloc[i]
        ENTROPY_B365.append(match_entropy(row, 'B365'))
        ENTROPY_BW.append(match_entropy(row, 'BW'))
        ENTROPY_IW.append(match_entropy(row, 'IW'))
        ENTROPY_LB.append(match_entropy(row, 'LB'))
        ENTROPY_WH.append(match_entropy(row, 'WH'))
        
        h, d, a = match_entropies(row)
        ENTROPY_H.append(h)
        ENTROPY_D.append(d)
        ENTROPY_A.append(a)
    # add to playing_stats these four columns, so for every match are available the 
    # cumulative values of goals scored or conceded for the teams involved in the match
    playing_stat['ENTROPY_B365'] = ENTROPY_B365
    playing_stat['ENTROPY_BW'] = ENTROPY_BW
    playing_stat['ENTROPY_IW'] = ENTROPY_IW
    playing_stat['ENTROPY_LB'] = ENTROPY_LB
    playing_stat['ENTROPY_WH'] = ENTROPY_WH
    playing_stat['ENTROPY_H'] = ENTROPY_H
    playing_stat['ENTROPY_D'] = ENTROPY_D
    playing_stat['ENTROPY_A'] = ENTROPY_A
    return playing_stat
    

# Fill data of matches without a team formation
def fill_teams_pts(playing_stat):
    for index, row in playing_stat.iterrows():
        #print(row)
        if np.isnan(row.HomeAttack):
            #print(playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeAttack, np.mean( playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeAttack  ))
            playing_stat.set_value(index, 'HomeAttack' , np.mean( playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeAttack  ) )
            playing_stat.set_value(index, 'HomeMedium' , np.mean( playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeMedium  ) )
            playing_stat.set_value(index, 'HomeDefense', np.mean( playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeDefense ) )
            playing_stat.set_value(index, 'HomeGK'     , np.mean( playing_stat[ playing_stat.HomeTeam == row.HomeTeam ].HomeGK      ) )
            playing_stat.set_value(index, 'AwayAttack' , np.mean( playing_stat[ playing_stat.AwayTeam == row.AwayTeam ].AwayAttack  ) )
            playing_stat.set_value(index, 'AwayMedium' , np.mean( playing_stat[ playing_stat.AwayTeam == row.AwayTeam ].AwayMedium  ) )
            playing_stat.set_value(index, 'AwayDefense', np.mean( playing_stat[ playing_stat.AwayTeam == row.AwayTeam ].AwayDefense ) )
            playing_stat.set_value(index, 'AwayGK'     , np.mean( playing_stat[ playing_stat.AwayTeam == row.AwayTeam ].AwayGK      ) ) 
            
        #print('____')
    #print(playing_stat[['HomeTeam', 'HomeAttack', 'HomeMedium']])    
    return playing_stat            
    
    
# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum
    
# Pattern identification
# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0

# to simplify the problem try only to guess if the home team is going to win
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'    
   
# calculate aggretated scores for team's playing areas     
def get_team_scores(team):
    coords = np.sort(team.coordinate.unique() ) # sorted from minor (goal keeper) to max (attack)
    #print(coords)
    team_score = np.round( sum(team.score), 3 )
    attack_score = 0
    medium_score = 0
    defense_score = 0
    gk_score = 0
    
    for i in range(len(coords)):
        s = sum( team.score[ team.coordinate == coords[i] ] )
        if i == 0:
            gk_score = s
        elif i == 1:
            defense_score = s
        elif i == 2:
            medium_score = s
        else:
            attack_score += s
                
    check = np.round( attack_score + medium_score + defense_score + gk_score, 3 ) 
    assert (team_score == check), "[ERROR] Team's score: calculated %f but should be %f"%(check, team_score)

    return attack_score, medium_score, defense_score, gk_score
    
    
def aggregate_scores(matches):
    # necessary because the data download from whoscored has different names for the teams
    teams_dict = {
        'AC Milan': 'Milan',
        'Robur Siena': 'Siena',
        'SPAL': 'Spal'
    }
    
    teams_scores = pd.DataFrame(columns = ['Date', 'HomeTeam', 'AwayTeam', 'HomeAttack', 'HomeMedium', 'HomeDefense', 'HomeGK', 'AwayAttack', 'AwayMedium', 'AwayDefense', 'AwayGK'])

    ht = None
    at = None
    for index, row in matches.iterrows():
        if row.HomeTeam != ht or row.AwayTeam != at:
            scores = matches[(matches.HomeTeam == row.HomeTeam) & (matches.AwayTeam == row.AwayTeam)]
        
            home_team = scores[ scores.team.str.contains('home') ]
            HomeAttack, HomeMedium, HomeDefense, HomeGK = get_team_scores(home_team)
        
            away_team = scores[ scores.team.str.contains('away') ]
            AwayAttack, AwayMedium, AwayDefense, AwayGK = get_team_scores(away_team)
        
            ht_name = home_team.HomeTeam.iloc[0]
            at_name = away_team.AwayTeam.iloc[0]
            # uniform names with those on other databases
            if ht_name in teams_dict.keys():
                ht_name = teams_dict[ht_name]
            if at_name in teams_dict.keys():
                at_name = teams_dict[at_name]
            s = pd.DataFrame.from_dict([{
                'Date' : home_team.date.iloc[0],
                'HomeTeam' : ht_name,
                'AwayTeam' : at_name,
                'HomeAttack' : HomeAttack,
                'HomeMedium' : HomeMedium,
                'HomeDefense' : HomeDefense,
                'HomeGK' : HomeGK,
                'AwayAttack' : AwayAttack,
                'AwayMedium' : AwayMedium,
                'AwayDefense' : AwayDefense,
                'AwayGK' : AwayGK}])
            #print('---')
            #print(s.head)
            # [home_team.date.iloc[0], ht_name, at_name, HomeAttack, HomeMedium, HomeDefense, HomeGK, AwayAttack, AwayMedium, AwayDefense, AwayGK]
        
            teams_scores = teams_scores.append(s, ignore_index=True)
            ht = row.HomeTeam
            at = row.AwayTeam
            
    return teams_scores
    
def  compute_team_scores(playing_stat, scores):
    teams_scores = aggregate_scores(scores)
    
    HomeAttack  = []
    HomeMedium  = []
    HomeDefense = []
    HomeGK      = []
    AwayAttack  = []
    AwayMedium  = []
    AwayDefense = []
    AwayGK      = []
    for index, row in playing_stat.iterrows():
        match = teams_scores[(teams_scores.HomeTeam == row.HomeTeam) & (teams_scores.AwayTeam == row.AwayTeam)]
        if not match.empty:
            HomeAttack.append( float(match.HomeAttack ))
            HomeMedium.append( float(match.HomeMedium ))
            HomeDefense.append( float(match.HomeDefense ))
            HomeGK.append( float(match.HomeGK ))
            AwayAttack.append( float(match.AwayAttack ))
            AwayMedium.append( float(match.AwayMedium ))
            AwayDefense.append( float(match.AwayDefense ))
            AwayGK.append( float(match.AwayGK ))
        else:
            print('[WARNING] - whoscored.com has not match: ', row)
            HomeAttack.append( None )
            HomeMedium.append( None )
            HomeDefense.append( None )
            HomeGK.append( None )
            AwayAttack.append( None )
            AwayMedium.append( None )
            AwayDefense.append( None )
            AwayGK.append( None )
        
    playing_stat['HomeAttack']  = HomeAttack
    playing_stat['HomeMedium']  = HomeMedium
    playing_stat['HomeDefense'] = HomeDefense
    playing_stat['HomeGK']      = HomeGK
    playing_stat['AwayAttack']  = AwayAttack
    playing_stat['AwayMedium']  = AwayMedium
    playing_stat['AwayDefense'] = AwayDefense
    playing_stat['AwayGK']      = AwayGK

    return playing_stat
    
    