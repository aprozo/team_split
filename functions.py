import pandas as pd
import numpy as np
from tensorflow import keras
import streamlit as st

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the model (only executed once!)
@st.cache_resource
def load_model():
	  return keras.models.load_model('model.keras')

all_players = ['Бот','Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
input_NN_line = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]

columnsTeam1 = ['Player1','Player2','Player3','Player4']
columnsTeam2 = ['Player5','Player6','Player7','Player8']
columnsPlayers = columnsTeam1 + columnsTeam2


map_weights = {
'ливень':5,
'кровавая жатва':5, 
'переход':3,
'мрачный карнавал':5,
'холодный ручей':4,
'похоронный звон':5,
'приход':5,
'смерть в воздухе':5,
'нет милосердию':5,
'вымерший центр':4,
'болотная лихорадка':4,
'последний рубеж':2,
'жертва':3,
'роковой полет':5}

def read_data( timeStart='', useMapWeight=False):
    data=pd.read_csv("datasetNN.csv")    # preprocess the data
    # Date,Map,Player1,Player2,Player3,Player4,Player5,Player6,Player7,Player8,TeamWon
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    if not useMapWeight:
        data["Map_weight"] =np.ones(len(data)) # remove map weight
    else:
        data["Map_weight"] = data["Map"].apply( lambda x: map_weights[x])
        data["Map_weight"] = data["Map_weight"]/ data["Map_weight"].max()
    #  drop entries which were played before the start date
    if timeStart != '':
        data = data[data['Date'] >= timeStart]
    
    data = data[data[columnsPlayers].apply(lambda x: all([player in all_players for player in x]), axis=1)]
    #  reset the index of data
    data.reset_index(drop=True, inplace=True)

    #  create a row with all the players and for each player, if he is in team1 or team2, and add TeamWon
    input_nn = pd.DataFrame(columns=input_NN_line)
    for i in range(0, len(data)):
        row = np.zeros(len(input_NN_line))
        for player in all_players:
            if player in data.iloc[i][columnsTeam1].values:
                row[input_NN_line.index(player+"_team1")] = 1
            if player in data.iloc[i][columnsTeam2].values:
                row[input_NN_line.index(player+"_team2")] = 1
        input_nn.loc[len(input_nn)] = row

    input_nn['TeamWon'] = data['TeamWon']
    if useMapWeight:
        input_nn['Map_weight'] = data['Map_weight']
    return input_nn


def train_model(test_size=0.2,dropout_rate=0.2, nEpochs = 100, useMapWeight=False, timeStart='', add_swap=False):

    input_nn=read_data(timeStart, useMapWeight)
    # split the data into training and testing
    input_nn = input_nn.sample(frac=1).reset_index(drop=True)
    X = input_nn.drop(columns=["TeamWon"])
    y = input_nn["TeamWon"]-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    #  add augumented data with swapped players for symmetrical training
    if add_swap:
        X_train_swap = X_train.copy()
        for player in all_players:
            X_train_swap[player+"_team1"], X_train_swap[player+"_team2"] = X_train_swap[player+"_team2"], X_train_swap[player+"_team1"]
        y_train_swap=1-y_train
        X_train = pd.concat([X_train, X_train_swap], ignore_index=True)
        y_train = pd.concat([y_train, y_train_swap], ignore_index=True)


    if useMapWeight: 
        w_train = X_train['Map_weight']
        w_test = X_test['Map_weight']
        X_train = X_train.drop(columns=["Map_weight"])
        X_test = X_test.drop(columns=["Map_weight"])

    # create the model
#  create a model
    model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(len(input_NN_line),), activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(dropout_rate), 
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(1, activation='sigmoid')
    ])
    from keras.callbacks import EarlyStopping 
    # Early stopping criteria to avoid overtraining
    earlystopping = EarlyStopping(patience=20,
                                verbose=1,
                                restore_best_weights=False)

    # compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics= [] if useMapWeight else ['accuracy'],
                weighted_metrics = ['accuracy'] if useMapWeight else []
                )
    
    # train the model
    model.fit(X_train,
            y_train, 
            batch_size=32,
            sample_weight=w_train if useMapWeight else None,
            validation_data=(X_test, y_test, w_test if useMapWeight else None), 
            # callbacks=[earlystopping],
            epochs=nEpochs)
    

    accuracy = model.evaluate(X_test, y_test)[1]

    return accuracy 


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

    teamwon = 1
    if np.random.rand() > 0.5:
        team1 = ','.join(names_win)
        team2 = ','.join(names_lose)
        teamwon = 1
    else:
        team1 = ','.join(names_lose)
        team2 = ','.join(names_win)
        teamwon = 2
        
    new_line = f'{date},{campaign},{team1},{team2}, {teamwon}\n'
    with open('datasetNN.csv', 'a') as file:
        file.write(new_line)
    print('Added to datasetNN.csv: ', new_line) 
    return True