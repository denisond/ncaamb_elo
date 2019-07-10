import numpy as np
import pandas as pd
import statistics as stats
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set()


from datetime import timedelta
from datetime import datetime

import plotly.plotly as py
import plotly.graph_objs as go 
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls





reg_szn = pd.read_csv("data/RegularSeasonCompactResults.csv")
# reg_szn.describe()



# Looks like we have match-level data from 1985 to 2018

# Select a unique list of team IDs
team_ids = set(reg_szn.WTeamID).union(set(reg_szn.LTeamID))

# Elo updates will be scaled based on the margin of victory
reg_szn['margin'] = reg_szn.WScore - reg_szn.LScore

def update_elos(rs=reg_szn,K = 20, HOME_ADVANTAGE = 100):
    """ Elo Rating Function.
    
    Args: 
        rs (pandas DataFrame): DataFrame of regular season match-level data.
        K  (int)             : Determines sensitivity of updates to Elo ratings.
        HOME_ADVANTAGE (int) : Points added/from teams' Elo ratings before prediction and updates.


    Returns:
        DataFrame with one observation per match with predictions and Elo rating columns.

    """
    rs = rs.copy()
    elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))  # Create baseline Elo ratings for each team


    def elo_pred(elo1, elo2):
        return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.)) # Use logistic CDF to predict match outcome
    
    def expected_margin(elo_diff):
        return((7.5 + 0.006 * elo_diff)) # Margin of victory multiplier, implemented by FiveThirtyEight

    def elo_update(w_elo, l_elo, margin): # Update Elo scores
        elo_diff = w_elo - l_elo
        pred = elo_pred(w_elo, l_elo)
        mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
        update = K * mult * (1 - pred)
        return(pred, update)

    preds = []
    w_elo = []
    l_elo = []

    # Loop over all rows of the games dataframe
    for row in rs.itertuples():
        # Get key data from current row
        w = row.WTeamID
        l = row.LTeamID
        margin = row.margin
        wloc = row.WLoc

        # Does either team get a home-court advantage?
        w_ad, l_ad, = 0., 0.
        if wloc == "H":
            w_ad += HOME_ADVANTAGE
        elif wloc == "A":
            l_ad += HOME_ADVANTAGE

        # Get elo updates as a result of the game
        pred, update = elo_update(elo_dict[w] + w_ad,
                                  elo_dict[l] + l_ad, 
                                  margin)
        elo_dict[w] += update
        elo_dict[l] -= update

        # Save prediction and new Elos for each round
        preds.append(pred)
        w_elo.append(elo_dict[w])
        l_elo.append(elo_dict[l])
    rs['preds'] = preds
    rs['w_elo'] = w_elo
    rs['l_elo'] = l_elo

    
    return rs, np.mean(-np.log(preds))

df = pd.DataFrame(update_elos()[0])


# Below we are going to restructure our data from one match per row to one match per team per row.


df_new = pd.DataFrame()
for team in team_ids:
    df_temp_w = df.loc[(df['WTeamID'] == team)]
    df_temp_w.rename(columns ={'WTeamID':'TeamID', 'w_elo':'elo'},inplace = True)
    df_temp_w = df_temp_w[['Season', 'DayNum', 'TeamID','elo']] 

    df_temp_l = df.loc[(df['LTeamID'] == team)]
    df_temp_l.rename(columns ={'LTeamID':'TeamID', 'l_elo':'elo'},inplace = True)
    df_temp_l = df_temp_l[['Season', 'DayNum', 'TeamID','elo']]

    df_temp = pd.concat([df_temp_w,df_temp_l])
    df_new = pd.concat([df_new,df_temp])
    
df_new.sort_values(['Season','DayNum'],inplace = True)

# We can get the start date of each season from the dataset below, which we will use to create a datetime index for our main dataset

seasons = pd.read_csv("data/Seasons.csv")
seasons = seasons[['Season','DayZero']]


df_w_seasons = pd.merge(df_new, seasons, how='left',on= 'Season')
df_w_seasons['date_temp'] = pd.to_datetime(df_w_seasons['DayZero'])
df_w_seasons['DayNum'] = df_w_seasons['DayNum'].apply(lambda x: int(x))


# Define function to increment start date based on DayNum columns (days from begginning of season)
def func(df):
    return df['date_temp']+ timedelta(days=df['DayNum'])
df_w_seasons['date'] = df_w_seasons[['date_temp', 'DayNum']].apply(func,axis =1)


# Read-in and merge team names and conferences

teams = pd.read_csv("data/Teams.csv")[['TeamID','TeamName']]
conf = pd.read_csv("data/TeamConferences.csv")


df_w_teams = pd.merge(df_w_seasons,teams, how="left", on= 'TeamID')
df_w_teams2 = pd.merge(df_w_teams,conf, how="left", on= ['TeamID','Season'])


df_final = df_w_teams2[['date','elo','TeamID','TeamName','ConfAbbrev','Season']].sort_values(['TeamName','date']).set_index('date')

df = df_final.copy()


df.to_csv('data/elo_rating.csv', encoding='utf-8')