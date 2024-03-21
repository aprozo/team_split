import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.stateful_button import button

from functions import get_teams, add_data

st.write("#  Left 4 Dead 2 team composition")
campaigns = {
    'ливень',
    'кровавая жатва', 
    'переход',
    'мрачный карнавал',
    'холодный ручей',
    'похоронный звон',
    'приход',
    'смерть в воздухе',
    'нет милосердию',
    'вымерший центр',
    'болотная лихорадка',
    'последний рубеж',
    'жертва',
    'роковой полет'}
all_players = ['Бот','Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
input_NN_line = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]
shown_players =all_players[1:]
# make the list of players from table above
old_method_dataset = pd.read_csv("old_method_dataset.csv")


col = st.columns(3)

# create a list of checkboxes with all players
check_players = []
for i in range(0, len(shown_players)):
    check_players.append(col[i%3].checkbox(shown_players[i]))
    
#  create a button "Submit"
submit_button = st.button("Submit")

if submit_button:
    if check_players.count(True) > 8:
        st.error("You have selected more than 8 players", icon="🚨"
                 )
    if check_players.count(True) <= 4:
        st.error("You have selected less than 5 players", icon="🚨"
                 )
#  from check_players create a list of selected players
    present_players = [shown_players[i] for i in range(0, len(shown_players)) if check_players[i]]
    st.write("You have selected the following players:")
    st.write( [player for player in present_players])

    results_old, results_nn = get_teams (present_players)
    middle_nn = len(results_nn)//2
    middle_old = len(results_old)//2

    # find closest result to 0.5 in NN
    closest = results_nn.iloc[(results_nn["NN output"]-0.5).abs().argsort()[:1]]
    #  find the middle entry in the nn
    median = results_nn.iloc [[middle_nn]]
    median2 = results_nn.iloc [[middle_nn+1]]

    # concat previous dataframes
    nn_output = pd.concat([closest, median, median2])
    # set indeces of nn_output to "Best","Median","Median2"
    nn_output.index = ["Best","Median","Median2"]
    # round the values in nn_output ans show percentage
    nn_output["NN output"] = (nn_output["NN output"]*100).round(2).astype(str) + '%'

    #reformat the list of string 	["Ден","Миша","Ондрей"] to "Ден - Миша - Ондрей"
    def get_nice_names(team):
        return " - ".join(team)
    nn_output["Team1"] = nn_output["Team1"].apply(get_nice_names)
    nn_output["Team2"] = nn_output["Team2"].apply(get_nice_names)


# ["Team1", "Team1 Winrate", "Team2", "Team2 Winrate"]
    # find closest result to 0.5 in old method
    closest_old = results_old.iloc[[middle_old]]
    closest2_old = results_old.iloc[[middle_old+1]]

    old_method_output = pd.concat([closest_old, closest2_old])
    old_method_output["Team1"] = old_method_output["Team1"].apply(get_nice_names)
    old_method_output["Team2"] = old_method_output["Team2"].apply(get_nice_names)



    old_method_output.index = ["Best","Second Best"]

    old_method_output["Team1 Winrate"] = old_method_output["Team1 Winrate"].round(2)
    old_method_output["Team2 Winrate"] = old_method_output["Team2 Winrate"].round(2)

    # create a column with a string result of teams winrate
    old_method_output["Comment"] = old_method_output.apply(lambda row: f"{100*row['Team1 Winrate']}%  vs  {100*row['Team2 Winrate']}%", axis=1)
    # drop the columns with winrates
    old_method_output = old_method_output.drop(columns=["Team1 Winrate", "Team2 Winrate"])

         
    table = st.table(nn_output)
    st.write("NN output is a probability of Team1 winning. The closer to 50 % the more balanced the teams are.")
    

    table = st.table(old_method_output)

    # display 

st.write("## ")

st.write("## Add a game to the dataset")
# create a list of checkboxes with players Team1 and Team2
# def add_data(names_win, names_lose, map='Map', date = ''):

check_players_team1 = []
check_players_team2 = []

shown_players1 = shown_players


st.write("### Team 1 (win)")
col1 = st.columns(6)
for i in range(0, len(shown_players)):
    check_players_team1.append(col1[i%6].checkbox(shown_players1[i], key=20+i))

left_players = shown_players


if button("Win team is ready", key="team1_button"):
    if check_players_team1.count(True) > 4:
        st.error("You have selected more than 4 players in the winning team", icon="🚨")
        st.stop()
    if check_players_team1.count(True) ==0:
        st.error("You have not selected any players in the winning team", icon="🚨")
        st.stop()
    left_players = [player for player in shown_players1 if player not in [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]]
    
    
    st.write("### Team 2 (lose)")
    col2 = st.columns(4)
    for i in range(0, len(left_players)):
        check_players_team2.append(col2[i%4].checkbox(left_players[i], key=100+i))

from datetime import date
today = date.today()
date = st.text_input("Date (YYYY-MM-DD)", value = today)
    
campaign = st.selectbox("Campaign", list(campaigns))

add_game_button = st.button("Add game to the dataset")
if add_game_button:
    if check_players_team2.count(True) > 4:
        st.error("You have selected more than 4 players in the losing team", icon="🚨")
        st.stop()

    if check_players_team2.count(True) ==0:
        st.error("You have not selected any players in the losing team", icon="🚨")
        st.stop()
    present_players_team1 = [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]
    left_players = [player for player in shown_players1 if player not in [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]]
    present_players_team2 = [left_players[i] for i in range(0, len(left_players)) if check_players_team2[i]]

    if add_data(present_players_team1, present_players_team2, campaign, date):
        st.success("Game added to the dataset")
        st.write("Team 1 (win):", present_players_team1)
        st.write("Team 2 (lose):", present_players_team2)

   

