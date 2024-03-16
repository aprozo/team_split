import streamlit as st
import pandas as pd
import joblib
import numpy as np

import keras
model = keras.models.load_model('model.keras')

st.write("#  Left 4 Dead 2 team composition")
all_players = ['Бот','Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
input_data_players = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]
# make the list of players from table above
old_method_dataset = pd.DataFrame({
    "Player": ["Сеньор", "Гриша", "Миша", "Ваня", "Бот", "Джун", "Данил", "Вадим", "Ондрей", "Юран", "Ден", "Савва"],
    "Games":  [      21,      22,     19,     18,     6,      8,      16,      24,       12,     10,    18,      10],
    "Wins":   [      14,      14,     11,     10,     3,      4,       8,      10,        5,      4,     6,       3]
})

old_method_dataset["Winrate"] = old_method_dataset["Wins"] / old_method_dataset["Games"]
old_method_dataset = old_method_dataset.sort_values(by="Winrate", ascending=False)

col = st.columns(3)

# create a list of checkboxes with all players
check_players = []
for i in range(0, len(all_players)):
    check_players.append(col[i%3].checkbox(all_players[i]))
    
#  create a button "Submit"
submit_button = st.button("Submit")

def winrate_oldmethod_output(present_players):
    all_teams = []
    for i in range(0, len(present_players)):
        for j in range(i+1, len(present_players)):
            for k in range(j+1, len(present_players)):
                for l in range(k+1, len(present_players)):
                    team1 = [present_players[i], present_players[j], present_players[k], present_players[l]]
                    team2 = [x for x in present_players if x not in team1]
                    while len(team2) < 4:
                        team2.append("Бот")
                    team1_winrate = old_method_dataset[old_method_dataset["Player"].isin(team1)]["Winrate"].sum()
                    team2_winrate = old_method_dataset[old_method_dataset["Player"].isin(team2)]["Winrate"].sum()
                    all_teams.append((team1, team1_winrate, team2, team2_winrate))
    all_teams.sort(key=lambda x: x[1], reverse=True)
    return all_teams


def nn_output(present_players):
    all_results = []
    # add bots to the present players if there are less than 8 players
    while len(present_players) < 8:
        present_players = present_players + ["Бот"]

    for i in range(0, len(present_players)):
        for j in range(i+1, len(present_players)):
            for k in range(j+1, len(present_players)):
                for l in range(k+1, len(present_players)):
                                    team1 = [present_players[i], present_players[j], present_players[k], present_players[l]]
                                    # subtract team 1 player form list of present players, "Бот" can be used many times
                                    team2 = [x for x in present_players if x not in team1]
                                    # if team 2 has less than 4 players, add "Бот" to team 2
                                    while len(team2) < 4:
                                        team2 = team2 + ["Бот"]
                                    # create a row with the players
                                    row = np.zeros(len(input_data_players))
                                    for player in team1:
                                        row[input_data_players.index(player+"_team1")] = 1
                                    for player in team2:
                                        row[input_data_players.index(player+"_team2")] = 1
                                    row = row.reshape(1, -1)
                                    # predict the result
                                    result = model.predict(row, verbose=0)
                                    all_results.append((team1, team2, result))
    # sort the results
    all_results.sort(key=lambda x: x[2], reverse=True)
    return all_results

output_dataframe = pd.DataFrame(columns=["Team 1", "Team 2", "Comment"])

if submit_button:
    
#  from check_players create a list of selected players
    selected_players = [all_players[i] for i in range(0, len(all_players)) if check_players[i]]


    st.write("You have selected the following players:")
 
    st.write( [player for player in selected_players])

    nn_results = nn_output(selected_players)
    old_results = winrate_oldmethod_output(selected_players)
    
    middle_nn = len(nn_results) // 2
    middle_old = len(old_results) // 2

    closest = min(nn_results, key=lambda x: abs(x[2] - 0.5))
    median = nn_results[middle_nn]
    median2 = nn_results[middle_nn-1]

    old = old_results[middle_old]
    old2 = old_results[middle_old-2]

    output_dataframe = pd.DataFrame({
        "Team 1": [closest[0], median[0], median2[0], old[0], old2[0]],
        "Team 2": [closest[1], median[1], median2[1], old[2], old2[2]],
        "Comment": [str(round(1+closest[2][0][0],2)), str(round(1+median[2][0][0],2)), str(round(1+median2[2][0][0],2)), str(round(old[1], 2)) + " vs " + str(round(old[3], 2)), str(round(old2[1], 2)) + " vs " + str(round(old2[3], 2))]},
        index=["NN_best", "NN_median", "NN_median2", "Old_best", "Old_best2"])

         
table = st.table(output_dataframe)
    

