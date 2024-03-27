import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.stateful_button import button
from functions import get_teams, add_data, train_model
from streamlit_image_select import image_select
from PIL import Image

st.image("l4d2.png")
st.write("#  Left 4 Dead 2 team composition")

st.sidebar.markdown('''
# Navigation
- [Team Split](#select-players-for-the-game)
- [Winrate of the players](#winrate-of-the-players)
- [History](#history)
- [Add a game](#add-a-game)
- [Train the model](#train-the-model)
''', unsafe_allow_html=True)



all_players = ['Бот','Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
input_NN_line = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]
shown_players =all_players[1:]

file_names= ["Blood_Harvest.webp", "Crash_Course.webp", "DeadCenter.webp", "Death_Toll.webp", "It's_Your_Funeral.webp", 
                "TheNewParish.webp", "The_Last_Stand.webp", "COLDSTEAMPOSTER.webp",  
                "Dead_Air.webp", "HardRain.webp", "No_Mercy.webp", "TheNewSwampFever.JPG.webp", "The_Passing_Poster.webp","Dark_Carnival02.webp"]

campaigns = [
        'кровавая жатва',
        'роковой полет',
        'вымерший центр',
        'похоронный звон',
        'жертва',
        'приход',
        'последний рубеж',
        'холодный ручей',
        'смерть в воздухе',
        'ливень',
        'нет милосердию',
        'болотная лихорадка',
        'переход',
        'мрачный карнавал'
        ]


@st.cache_data
def load_data():
    return pd.read_csv("old_method_dataset.csv")

@st.cache_data
def load_images():
    return  [Image.open("./Campaigns/"+file_name) for file_name in file_names]

st.header("Team Split")
with st.form(key="my_form"):
    st.write("### Select players for the game")
    col = st.columns(3)
    # create a list of checkboxes with all players
    check_players = []
    for i in range(0, len(shown_players)):
        check_players.append(col[i%3].checkbox(shown_players[i]))
    #  create a button "Submit"
    submit_button =  st.form_submit_button("Submit")

if submit_button:
    if check_players.count(True) > 8:
        st.error("You have selected more than 8 players", icon="🚨")
        st.stop()
    if check_players.count(True) <= 4:
        st.error("You have selected less than 5 players", icon="🚨")
        st.stop()
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

    st.write("### Neural Network")     
    table = st.table(nn_output)
    st.write("NN output is a probability of Team1 winning. The closer to 50 % the more balanced the teams are.")
    
    st.write("### Winrate sum method")     
    table = st.table(old_method_output)

    # display 
st.write("---")
st.header("Winrate of the players")


if button("Show winrate", key="show_winrate"):
    # format the column winrate to show percentage
    old_method_dataset= load_data()
    st.write(old_method_dataset.style.background_gradient( gmap=old_method_dataset['Winrate'], cmap='RdYlGn',vmin=0,vmax=1,axis=0).to_html(), unsafe_allow_html=True)

    # display 
st.write("---")
st.header("History")

if button("Show history", key="show_history"):

    # get the last 5 games from the dataset
    history = pd.read_csv("datasetNN.csv")
    ngames=st.slider("Number of games displayed", 1,len(pd.read_csv("datasetNN.csv")), 5, 1)
    lastgames = history.tail(ngames).iloc[::-1]
    lastgames['Team1'] = lastgames[lastgames.columns[2:6]].apply(
        lambda x: ','.join(x.dropna().astype(str)),axis=1)
    lastgames['Team2'] = lastgames[lastgames.columns[6:10]].apply(
        lambda x: ','.join(x.dropna().astype(str)),axis=1)
    lastgames = lastgames.drop(columns=lastgames.columns[2:10])

    def highlight(s):
        color2 = f"background-color:lightcoral"
        color = f"background-color:lightgreen" 
        #condition
        cond = s['TeamWon'] == 1
        # DataFrame of styles
        df = pd.DataFrame('', index=s.index, columns=s.columns)
        # set columns by condition
        df.loc[cond, 'Team1'] = color
        df.loc[~cond, 'Team1'] = color2
        # if not m, set another color
        df.loc[cond, 'Team2'] = color2
        df.loc[~cond, 'Team2'] = color
        return df

    styler = lastgames.style.apply(highlight, axis=None).hide('TeamWon', axis=1)
    st.write(styler.to_html(escape=False), unsafe_allow_html=True)

st.write("---")
st.header("Add a game")

if button("Add a game", key="show_add_game"):
    # create a list of checkboxes with players Team1 and Team2
    # def add_data(names_win, names_lose, map='Map', date = ''):

    check_players_team1 = []
    check_players_team2 = []

    shown_players1 = shown_players

    def is_size_ok(team):
        if team.count(True) > 4:
            st.error("You have selected more than 4 players in the team", icon="🚨")
            st.stop()
        if team.count(True) ==0:
            st.error("You have not selected any players in the team", icon="🚨")
            st.stop()

    st.write("### Team (win) - select 1-4 players")
    col1 = st.columns(6)
    for i in range(0, len(shown_players)):
        check_players_team1.append(col1[i%6].checkbox(shown_players1[i], key=20+i))

    if button("Win team is ready", key="team1_button"):
        is_size_ok(check_players_team1)
        left_players = [player for player in shown_players1 if player not in [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]]
        
        with st.form(key="my_form3"):
            st.write("### Team (lose)")
            col2 = st.columns(4)
            for i in range(0, len(left_players)):
                check_players_team2.append(col2[i%4].checkbox(left_players[i], key=100+i))

            from datetime import date
            date = st.date_input("Date", value = date.today(), min_value = date(2022, 10, 28), max_value = date.today())
            
        
            ind = image_select(
            label="Select a Campaign",
            images=load_images(),
            captions= campaigns,
            return_value="index",
            use_container_width=False
            )
            campaign = campaigns[ind]
            submit_button_ =  st.form_submit_button("Add game to the dataset")
            if submit_button_:
                st.write("### Selected campaign:", campaign)
                is_size_ok(check_players_team2)
                present_players_team1 = [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]
                left_players = [player for player in shown_players1 if player not in [shown_players1[i] for i in range(0, len(shown_players1)) if check_players_team1[i]]]
                present_players_team2 = [left_players[i] for i in range(0, len(left_players)) if check_players_team2[i]]
                
                if add_data(present_players_team1, present_players_team2, campaign, date):
                    st.success("Game a added to the dataset")
                    st.write("Team (win):", present_players_team1)
                    st.write("Team (lose):", present_players_team2)
                    st.write("Date:", date)
                    st.write("Campaign:", campaign)

    # display 
st.write("---")
st.header("Train the model")

from datetime import date
if  button("Train the model", key="show_retrain"):
    with st.form(key="my_form2"):
        st.write("### Train the model")
        st.write("The model will be retrained with the new data")
        # creat tuning of parameters
        test_size = st.slider("Test size (  % of the dataset used for testing)",  0.05, 0.5, 0.1, step=0.05)
        dropout_rate = st.slider("Dropout rate (avoid overfitting) ", 0., 0.5, 0.1, step=0.05)
        nEpochs = st.slider("Number of epochs (how many times the model will use the data)" , 10, 200, 100, step=10)
        useMapWeight = st.checkbox("Use campaign weight (campaigns with more maps will have more weight)")
        add_swap = st.checkbox("Add swapped teams for symmetry training", value = True)         #(2022, 10, 28) - first match
        timeStart = st.date_input("Start date of the first game (include the games after this date into the training set )", value = date(2022, 10, 28), min_value = date(2022, 10, 28), max_value = date.today())

        submit_button_retrain =st.form_submit_button("train")

        if submit_button_retrain:
            with st.spinner('Training'):
                accuracy=train_model(test_size, dropout_rate, nEpochs, useMapWeight, timeStart, add_swap)
                accuracy = round(accuracy, 2)
            st.write("### Accuracy" , accuracy)
            st.balloons()
            st.success("The model is retrained")