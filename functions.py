import pandas as pd
import numpy as np
from tensorflow import keras
import streamlit as st

# Load the model (only executed once!)
@st.cache_resource
def load_model():
	  return keras.models.load_model('model.keras')


def get_teams(present_players):
    if len(present_players)>8: 
        return "Too many players"
    old_method_dataset = pd.read_csv("old_method_dataset.csv")
    all_teams_old_method = pd.DataFrame(columns=["Team1", "Team1 Winrate", "Team2", "Team2 Winrate"])
    all_teams_nn = pd.DataFrame(columns=["Team1", "Team2","NN output"])

    all_players = ['Бот','Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
    input_NN_line = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]


    input_nn =pd.DataFrame(columns=input_NN_line)

    if len(present_players)%2==1:
        present_players = present_players + ["Бот"]  

    for i in range(0, len(present_players)):
        for j in range(i+1, len(present_players)):
            for k in range(j+1, len(present_players)):
                if len(present_players)==6:
                    team1 = [present_players[i], present_players[j], present_players[k]]
                    team2 = [x for x in present_players if x not in team1]
                    team1_winrate = old_method_dataset[old_method_dataset["Player"].isin(team1)]["Winrate"].sum()
                    team2_winrate = old_method_dataset[old_method_dataset["Player"].isin(team2)]["Winrate"].sum()
                    all_teams_old_method.loc[len(all_teams_old_method)] = [team1, team1_winrate, team2, team2_winrate]
                    # create a row with the players
                    row = np.zeros(len(input_NN_line))
                    for player in team1:
                        row[input_NN_line.index(player+"_team1")] = 1
                    for player in team2:
                        row[input_NN_line.index(player+"_team2")] = 1
                    # add row to teams_nn
                    input_nn.loc[len(input_nn)] = row
                    all_teams_nn.loc[len(all_teams_nn)] = [team1, team2, 0]
                if len(present_players)==8:
                    for l in range(k+1, len(present_players)):
                        team1 = [present_players[i], present_players[j], present_players[k], present_players[l]]
                        team2 = [x for x in present_players if x not in team1]
                        team1_winrate = old_method_dataset[old_method_dataset["Player"].isin(team1)]["Winrate"].sum()
                        team2_winrate = old_method_dataset[old_method_dataset["Player"].isin(team2)]["Winrate"].sum()
                        all_teams_old_method.loc[len(all_teams_old_method)] = [team1, team1_winrate, team2, team2_winrate]
                        # create a row with the players
                        row = np.zeros(len(input_NN_line))
                        for player in team1:
                            row[input_NN_line.index(player+"_team1")] = 1
                        for player in team2:
                            row[input_NN_line.index(player+"_team2")] = 1
                        # add row to teams_nn
                        input_nn.loc[len(input_nn)] = row
                        all_teams_nn.loc[len(all_teams_nn)] = [team1, team2, 0]
              

    all_teams_old_method.sort_values(by="Team1 Winrate", inplace=True)
    all_teams_old_method.reset_index(drop=True, inplace=True)

    # predict the output of the NN
    model = load_model()
    all_teams_nn["NN output"] = model.predict(input_nn)
    all_teams_nn.sort_values(by="NN output", inplace=True)

    return all_teams_old_method, all_teams_nn


def add_data(names_win, names_lose, campaign='Map', date = ''):
    old_method_dataset = pd.read_csv("old_method_dataset.csv")
    if len(names_win)>len(names_lose):
        names_lose += ['Бот']
    elif len(names_win)<len(names_lose):
        names_win += ['Бот']

    for name in names_win:
        old_method_dataset.loc[old_method_dataset['Player'] == name, 'Wins'] += 1
        old_method_dataset.loc[old_method_dataset['Player'] == name, 'Games'] += 1
    for name in names_lose:
        old_method_dataset.loc[old_method_dataset['Player'] == name, 'Games'] += 1
    old_method_dataset["Winrate"] = old_method_dataset["Wins"] / old_method_dataset["Games"]
    old_method_dataset = old_method_dataset.sort_values(by="Winrate", ascending=False)
    old_method_dataset["Winrate"]=old_method_dataset["Winrate"].apply(lambda x: round(x, 2))

    old_method_dataset.to_csv("old_method_dataset.csv", index=False)

    # columns - Date,Map,Player1,Player2,Player3,Player4,Player5,Player6,Player7,Player8,TeamWon
    # add to datasetNN a line with date, map, team1, team2, whichteamwon:
    # e.g. 2024-03-09,Map,Ден,Гриша,Ондрей,Юран,Сеньор,Савва,Данил,Бот,1.0

    while len(names_win)< 4:
        names_win += ['Бот']
    while len(names_lose)< 4:
        names_lose += ['Бот']
    team1 = ','.join(names_win)
    team2 = ','.join(names_lose)
    new_line = f'{date},{campaign},{team1},{team2},1\n'
    with open('datasetNN.csv', 'a') as file:
        file.write(new_line)
    print('Added to datasetNN.csv: ', new_line) 
    return True