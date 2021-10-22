
def team_details_count(path):
    team_details_dictionary = {}
    rf = open(path,'r')
    for line in rf.readlines():
        str_list = line.strip().split('\t')
        team1 = str_list[0].lower()
        team2 = str_list[1].lower()
        goal1 = float(str_list[2])
        goal2 = float(str_list[3])
        year = int(str_list[6])
        # count
        record_dict1 = {'win':0.0,'draw':0.0,'lose':0.0,'goal':0.0,'match':0.0}
        team_details_dictionary.setdefault(team1,record_dict1)
        record_dict2 = {'win':0.0,'draw':0.0,'lose':0.0,'goal':0.0,'match':0.0}
        team_details_dictionary.setdefault(team2,record_dict2)
        # win draw lose
        if(goal1>goal2):
            team_details_dictionary[team1]['win']+=1
            team_details_dictionary[team2]['lose']+=1
        elif(goal1<goal2):
            team_details_dictionary[team1]['lose']+=1
            team_details_dictionary[team2]['win']+=1
        else:
            team_details_dictionary[team1]['draw']+=1
            team_details_dictionary[team2]['draw']+=1
        # match
        team_details_dictionary[team1]['match']+=1
        team_details_dictionary[team2]['match']+=1
        # goal
        team_details_dictionary[team1]['goal']+=goal1
        team_details_dictionary[team2]['goal']+=goal2

    return team_details_dictionary


def return_team_details(team_details_dictionary, nation1):
    win1 = team_details_dictionary[nation1]['win']/team_details_dictionary[nation1]['match']
    draw1 = team_details_dictionary[nation1]['draw']/team_details_dictionary[nation1]['match']
    lose1 = team_details_dictionary[nation1]['lose']/team_details_dictionary[nation1]['match']
    goal1 = team_details_dictionary[nation1]['goal']/team_details_dictionary[nation1]['match']
    return [win1, draw1, lose1, goal1]


def read_training_data(history_path, is_knockout):
    team_details_dictionary = team_details_count(history_path)
    X = []
    y = []
    rf = open(history_path, 'r')
    for line in rf.readlines():
        str_list = line.strip().split('\t')
        team1 = str_list[0].lower()
        team2 = str_list[1].lower()
        # history record
        team1_detail = return_team_details(team_details_dictionary, team1)
        team2_detail = return_team_details(team_details_dictionary, team2)
        elo1 = float(str_list[4])/2000
        elo2 = float(str_list[5])/2000
        # gather together
        vec = [elo1, elo2]
        vec.extend(team1_detail)
        vec.extend(team2_detail)
        # save to X,y
        yi = int(str_list[2])-int(str_list[3])
        if is_knockout and yi == 0:
            continue
        X.append(vec)
        y.append(yi)
    return X, y


def read_euro2020(path):
    rf = open(path, 'r')
    rf.readline()
    team_details_dictionary = {}
    group_nation_dict = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': []}
    for line in rf.readlines():
        str_list = line.strip().split(',')
        # group-nation
        group_nation_dict.setdefault(str_list[0], []).append(str_list[2])
        # nation-info
        info_dict = {'group': '0', 'elo': 0}
        team_details_dictionary.setdefault(str_list[2], info_dict)
        team_details_dictionary[str_list[2]]['group'] = str_list[0]
        team_details_dictionary[str_list[2]]['elo'] = float(str_list[3]) / 2000
    return team_details_dictionary, group_nation_dict


def Model_Selector(model_info_path):

    handler = open(model_info_path, 'r')
    model_info = handler.readlines()
    details = model_info[0].split(",")
    return details