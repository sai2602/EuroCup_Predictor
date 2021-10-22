import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_loader.helper_functions import read_euro2020, team_details_count,return_team_details,Model_Selector, read_training_data


def run_group_stage(euro2020_groups_dictionary):
    promoted_file = open('./Results_Directory/Group_Stage_Promoted_nations.csv', 'w')
    group3rd_best_teams_dictionary = {}
    # top 2 teams in each group
    for group in euro2020_groups_dictionary.keys():
        count = 1
        for (team, point_and_goal) in euro2020_groups_dictionary[group]:
            if count > 3:
                continue
            if count == 3:
                group3rd_best_teams_dictionary.setdefault(group + str(count), (team, point_and_goal))
                continue
            promoted_file.write(
                group + str(count) + ',' + team + ',' + str(point_and_goal[0]) + ',' + str(point_and_goal[1]) + '\n')
            count += 1
    # best 4 3rd team in all groups
    group3rd_best_teams_dictionary = sorted(group3rd_best_teams_dictionary.items(), key=lambda d: d[1][1], reverse=True)
    count = 1
    for (group_name, (team, points_and_goal)) in group3rd_best_teams_dictionary:
        if count > 4:
            break
        promoted_file.write(
            group_name + ',' + team + ',' + str(points_and_goal[0]) + ',' + str(points_and_goal[1]) + '\n')
        count += 1
    promoted_file.close()


def total_points(team_list, result):
    team_goal_point_dictionary = {}
    for i in range(len(result)):
        print(team_list[i], result[i])
        team1 = team_list[i][0]
        team2 = team_list[i][1]
        team_goal_point_dictionary.setdefault(team1, [0, 0])
        team_goal_point_dictionary.setdefault(team2, [0, 0])
        team_goal_point_dictionary[team1][1] += result[i]
        team_goal_point_dictionary[team2][1] -= result[i]
        if result[i] > 0:
            team_goal_point_dictionary[team1][0] += 3
        elif result[i] < 0:
            team_goal_point_dictionary[team2][0] += 3
        else:
            team_goal_point_dictionary[team1][0] += 1
            team_goal_point_dictionary[team2][0] += 1
    # sort
    team_point_goal_dictionary = sorted(team_goal_point_dictionary.items(), key=lambda d: d[1], reverse=True)
    return team_point_goal_dictionary



if __name__ == '__main__':
    # train
    # load training data
    print("=========== WELCOME TO GROUP STAGE OF EURO 2020 ===============")
    print('loading training data...')
    history_path = './Data_Directory/Raw_data.txt'
    model_info_path = './Data_Directory/Model_Selector.txt'
    nation_record_dict = team_details_count(history_path)
    train_X, train_Y = read_training_data(history_path, False)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    # train
    print('Starting training with the training set...')

    model_details = Model_Selector(model_info_path)
    if model_details[0] == "XGB":
        print("Working with XGBoost classifier\n")
        score_model = XGBClassifier(n_estimators=int(model_details[1]), max_depth=int(model_details[2]))
    elif model_details[0] == "RFC":
        print("Working with RandomForest classifier\n")
        score_model = RandomForestClassifier(n_estimators=int(model_details[1]), max_depth=int(model_details[2]))
    else:
        raise AssertionError("Wrong model selected!!")

    # The mean log error
    score_model.fit(train_X, train_Y)
    Y_true = minmax_scale(train_Y, feature_range=(0, 1))
    Y_pred = minmax_scale(score_model.predict(train_X), feature_range=(0, 1))
    print("training set mean log error: %.5f" % mean_squared_log_error(Y_true, Y_pred))

    # predict
    # load prediction data
    print('Loading prediction data...')
    euro2020_path = './Data_Directory/Euro2020_Schedule.csv'
    team_details_dictionary, group_nation_dictionary = read_euro2020(euro2020_path)
    test_X = []
    vs_list = []
    for g in group_nation_dictionary.keys():
        for i in range(4):
            for j in range(i+1, 4):
                team1 = group_nation_dictionary[g][i]
                team2 = group_nation_dictionary[g][j]
                vs_list.append((team1, team2))
                team1_details = return_team_details(nation_record_dict, team1)
                team2_details = return_team_details(nation_record_dict, team2)
                elo1 = team_details_dictionary[team1]['elo']
                elo2 = team_details_dictionary[team2]['elo']
                vec = [elo1, elo2]
                vec.extend(team1_details)
                vec.extend(team2_details)
                # save all samples
                test_X.append(vec)
    # predict
    test_X = np.array(test_X)
    print('Predicting Results:')
    test_y = score_model.predict(test_X)

    # points count
    nation_point_goal_dict = total_points(vs_list,test_y)

    # group analysis and write it to a file
    group_sorted_dict = {}
    wf = open('./Results_Directory/Group_Stage_Point_Table.csv', 'w')
    for (team, points_and_goal) in nation_point_goal_dict:
        group = team_details_dictionary[team]['group']
        group_sorted_dict.setdefault(group, [])
        group_sorted_dict[group].append((team, points_and_goal))
        wf.write(group+','+team+','+str(points_and_goal[0])+','+str(points_and_goal[1])+'\n')
    wf.close()  
    
    # Run the group stage
    run_group_stage(group_sorted_dict)
