import numpy as np
from math import floor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import minmax_scale
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from data_loader.helper_functions import read_euro2020, team_details_count, return_team_details, Model_Selector, \
    read_training_data


Top_Best_Three_Teams_combination = ['abcd','abce','abcf','abde','abdf','abef','acde','acdf','acef',
                                    'adef','bcde','bcdf','bcef','bdef','cdef']


def get_match_winner(score_model, team1, team2):
    nation1_record = return_team_details(nation_record_dict, team1)
    nation2_record = return_team_details(nation_record_dict, team2)
    elo1 = team_details_dictionary[team1]['elo']
    elo2 = team_details_dictionary[team2]['elo']
    vec = [elo1, elo2]
    vec.extend(nation1_record)
    vec.extend(nation2_record)
    score = score_model.predict(np.array(vec).reshape(1, -1))
    return score


def get_team_id_dictionary(sorted_path):
    id_nation_dictionary = {}
    four_best_3rd = ''
    rf = open(sorted_path, 'r')
    for line in rf.readlines():
        str_list = line.strip().split(',')
        id_nation_dictionary.setdefault(str_list[0], str_list[1])
        if str_list[0][1] == '3':
            four_best_3rd += str_list[0][0]
    four_best3rd = ''.join((lambda x:(x.sort(),x)[1])(list(four_best_3rd)))
    return id_nation_dictionary, four_best3rd


def get_round_of_16_nation_list(id_path, top_four_3rd_place):
    round_of_16_nation_list = []
    rf = open(id_path, 'r')
    for line in rf.readlines():
        str_list = line.strip().split(',')
        if len(str_list) < 2:
            round_of_16_nation_list.append(line.strip())
        else:  # gourp 3rd
            round_of_16_nation_list.append(str_list[Top_Best_Three_Teams_combination.index(top_four_3rd_place)])
    return round_of_16_nation_list


def get_winner(low, high, circle):
    circle += 1
    if high-low < 2:
        # print(low, high)
        team1 = id_nation_dict[round_of_16_nation_list[low]]
        team2 = id_nation_dict[round_of_16_nation_list[high]]
        score = get_match_winner(score_model, team1, team2)
        print(pow(2, circle), (team1, team2), score)
        wf.write(str(pow(2, circle))+','+team1+','+team2+','+str(score[0])+'\n')
        if score >= 0:
            return low
        else:
            return high
    mid = floor((low+high)/2)
    win1_idx = get_winner(low, mid, circle)
    win2_idx = get_winner(mid+1, high, circle)
    team1 = id_nation_dict[round_of_16_nation_list[win1_idx]]
    team2 = id_nation_dict[round_of_16_nation_list[win2_idx]]
    score = get_match_winner(score_model, team1, team2)
    print(pow(2, circle), (team1, team2), score)
    wf.write(str(pow(2, circle))+','+team1+','+team2+','+str(score[0])+'\n')
    if score >= 0:
        return win1_idx
    else:
        return win2_idx


if __name__ == '__main__':

    # train
    # load training data
    print('Loading training data...')
    history_path = './Data_Directory/Raw_Data.txt'
    model_info_path = './Data_Directory/Model_Selector.txt'
    nation_record_dict = team_details_count(history_path)
    train_X, train_Y = read_training_data(history_path, True)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    # train
    print('start training...')

    model_details = Model_Selector(model_info_path)
    if model_details[0] == "XGB":
        score_model = XGBClassifier(n_estimators=int(model_details[1]), max_depth=int(model_details[2]))
    elif model_details[0] == "RFC":
        score_model = RandomForestClassifier(n_estimators=int(model_details[1]), max_depth=int(model_details[2]))
    else:
        raise AssertionError("Wrong model selected!!")

    score_model.fit(train_X, train_Y)
    Y_true = minmax_scale(train_Y, feature_range=(0, 1))
    Y_pred = minmax_scale(score_model.predict(train_X), feature_range=(0, 1))
    print("=========== WELCOME TO KNOCKOUT STAGE OF EURO 2020 ===============")
    print('trainset mean log error: %.9f' % mean_squared_log_error(Y_true, Y_pred))
    # predict
    # load prediction data
    print('Load the prediction data....')
    sorted_path = './Results_Directory/Group_Stage_Promoted_nations.csv'  # promoted teams
    id_nation_dict, four_best3rd = get_team_id_dictionary(sorted_path)
    id_path = './Data_Directory/Round_of_16.txt'  # round16 vs list
    round_of_16_nation_list = get_round_of_16_nation_list(id_path, four_best3rd)
    euro2016_path = './Data_Directory/Euro2020_Schedule.csv'  # euro2016 info
    team_details_dictionary, _ = read_euro2020(euro2016_path)
    # predict

    print('Knockout prediction results:')
    wf = open('./Results_Directory/Euro_2020_KnockOut_Results.csv', 'w')
    champion_id = get_winner(0, 15, 0)
    print("***************** THE WINNER OF EURO 2020 IS******************")
    print()
    print("*#*#*#*#*#*#*#*#", id_nation_dict[round_of_16_nation_list[int(champion_id)]].upper(), "*#*#*#*#*#*#*#*#",)
    wf.close()
