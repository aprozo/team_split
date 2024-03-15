import sys
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QMessageBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont

import pandas as pd
import numpy as np
import keras
model = keras.models.load_model('model.keras')

# based on present players and winrate, loop over all possible teams and calculate the total winrate
#  and print the best team

all_players = ['Бот', 'Вадим', 'Ваня', 'Гриша', 'Данил', 'Ден', 'Джун', 'Миша', 'Ондрей', 'Савва', 'Сеньор', 'Юран']
input_data_players = [player+"_team1" for player in all_players] + [player+"_team2" for player in all_players]

# make the list of players from table above
old_method_dataset = pd.DataFrame({
    "Player": ["Сеньор", "Гриша", "Миша", "Ваня", "Бот", "Джун", "Данил", "Вадим", "Ондрей", "Юран", "Ден", "Савва"],
    "Games":  [      21,      22,     19,     18,     6,      8,      16,      24,       12,     10,    18,      10],
    "Wins":   [      14,      14,     11,     10,     3,      4,       8,      10,        5,      4,     6,       3]
})

old_method_dataset["Winrate"] = old_method_dataset["Wins"] / old_method_dataset["Games"]
old_method_dataset = old_method_dataset.sort_values(by="Winrate", ascending=False)

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

# make gui with checkboxes for all players, by default all players are checked, a "Submit" button and output window where the prediction from pretrained model and old method will be printed

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Player Selection")
        self.setGeometry(100, 100, 800, 600)
        # increase default font size
        self.setFont(QFont("Arial", 15))

        # create layout for checboxes which is the table with 3 rows and 4 columns using GridLayout
        self.layout = QVBoxLayout()  # Change here
        self.setLayout(self.layout)  # Change here
        # Create a list to hold all the checkboxes
        self.checkboxes = []
        # Initialize the UI
        self.initUI()

    def initUI(self):
        # Create checkboxes for all players
        all_players = ["Сеньор", "Гриша", "Миша", "Ваня", "Джун", "Данил", "Вадим", "Ондрей", "Юран", "Ден", "Савва"]
        for player in all_players:
            checkbox = QCheckBox(player)
            # set  nice  font for the checkbox
            checkbox.setChecked(True)  # By default, all players are checked
            self.checkboxes.append(checkbox)
                
        # Add the checkboxes
        for i in range(0, len(self.checkboxes), 4):
            row = QHBoxLayout()
            for j in range(4):
                if i + j < len(self.checkboxes):
                    row.addWidget(self.checkboxes[i + j])
                # add empty widgets if there are less than 4 players
                else:
                    row.addWidget(QWidget())


            self.layout.addLayout(row)


        # Create a "Submit" button
        submit_button = QPushButton("Submit")
        # increase font size in the button
        submit_button.setFont(QFont("Arial", 20))
        # increase the button vertically
        submit_button.setFixedHeight(100)
        # apply color to the button
        submit_button.setStyleSheet("background-color : lightgreen")
        submit_button.clicked.connect(self.submit)
        # Create an 5 output windows with different results : NN_best, NN_second_best, NN_closest, Old_best, Old_second_best

        self.output_NN = QTableWidget() 
        self.output_NN.setColumnCount(3) # Method, team1, team2, output
        self.output_NN.setRowCount(5) # NN_best, NN_second_best, NN_closest, Old_best, Old_second_best
        self.output_NN.setHorizontalHeaderLabels([ "Team 1", "Team 2", "Comment"])
        self.output_NN.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.output_NN.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.output_NN.setVerticalHeaderLabels([ "NN_closest","NN_best", "NN_second_best", "Old_best", "Old_second_best"])

        # Add the widgets to the layout
        self.layout.addWidget(submit_button)
        self.layout.addWidget(self.output_NN)

        # Set the layout
        self.show()

    def submit(self):
        # Get the selected players
        selected_players = [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]
        print ("selected_players are", selected_players)
        
        # give warning qbox if more than 8 players are selected
        if len(selected_players) > 8:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Too many players selected")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return
      
        # Perform prediction using the pretrained model and old method
        nn_results = nn_output(selected_players)
        old_results = winrate_oldmethod_output(selected_players)
        closest = min(nn_results, key=lambda x: abs(x[2] - 0.5))
        middle_nn = len(nn_results) // 2
        middle_old = len(old_results) // 2

        self.output_NN.setItem(2, 0, QTableWidgetItem(str(closest[0])))
        self.output_NN.setItem(2, 1, QTableWidgetItem(str(closest[1])))
        self.output_NN.setItem(2, 2, QTableWidgetItem("Result of NN = "+str(round(1+closest[2][0][0],2))))

          # Print the results
        self.output_NN.setItem(0, 0, QTableWidgetItem(str(nn_results[middle_nn][0])))
        self.output_NN.setItem(0, 1, QTableWidgetItem(str(nn_results[middle_nn][1])))
        # write comment with "Result of NN = "
        self.output_NN.setItem(0, 2, QTableWidgetItem("Result of NN = "+str(round(1+nn_results[middle_nn][2][0][0],2))))
  
        self.output_NN.setItem(1, 0, QTableWidgetItem(str(nn_results[middle_nn-1][0])))
        self.output_NN.setItem(1, 1, QTableWidgetItem(str(nn_results[middle_nn-1][1])))
        self.output_NN.setItem(1, 2, QTableWidgetItem("Result of NN = "+str(round(1+nn_results[middle_nn-1][2][0][0],2))))


        self.output_NN.setItem(3, 0, QTableWidgetItem(str(old_results[middle_old][0])))
        self.output_NN.setItem(3, 1, QTableWidgetItem(str(old_results[middle_old][2])))
    # write comment with total winrate for both teams
        self.output_NN.setItem(3, 2, QTableWidgetItem(str(round(old_results[middle_old][1], 2)) + " vs " + str(round(old_results[middle_old][3], 2))))

        self.output_NN.setItem(4, 0, QTableWidgetItem(str(old_results[middle_old-2][0])))
        self.output_NN.setItem(4, 1, QTableWidgetItem(str(old_results[middle_old-2][2])))
        self.output_NN.setItem(4, 2, QTableWidgetItem(str(round(old_results[middle_old-2][1], 2)) + " vs " + str(round(old_results[middle_old-2][3], 2))))
      

def main():
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())

main()