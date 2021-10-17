import numpy as np
import data_loader.read_history_count as read_history_count
import data_loader.read_euro2016info as read_euro2016info
from math import floor

all_best3rd = ['abcd','abce','abcf','abde','abdf','abef','acde','acdf','acef',
               'adef','bcde','bcdf','bcef','bdef','cdef']


def read_id_nation_dict(sorted_path):
    id_nation_dict = {}
    four_best3rd = ''
    rf = open(sorted_path, 'r')
    for line in rf.readlines():
        str_list = line.strip().split(',')
        id_nation_dict.setdefault(str_list[0], str_list[1])
        if str_list[0][1] == '3':
            four_best3rd += str_list[0][0]
    four_best3rd = ''.join((lambda x:(x.sort(),x)[1])(list(four_best3rd)))
    return id_nation_dict, four_best3rd


def read_final16_nation_list(id_path, id_nation_dict, four_best3rd):
    final16_nation_list = []
    rf = open(id_path, 'r')
    for line in rf.readlines():
        str_list = line.strip().split(',')
        if len(str_list) < 2:
            final16_nation_list.append(line.strip())
        else:  # gourp 3rd
            final16_nation_list.append(str_list[all_best3rd.index(four_best3rd)])
    return final16_nation_list


def predict_match(score_model, nation1, nation2):
    nation1_record = read_history_count.get_nation1_record(nation_record_dict, nation1)
    nation2_record = read_history_count.get_nation1_record(nation_record_dict, nation2)
    elo1 = nation_info_dict[nation1]['elo']
    elo2 = nation_info_dict[nation2]['elo']
    vec = [elo1, elo2]
    vec.extend(nation1_record)
    vec.extend(nation2_record)
    score = score_model.predict(np.array(vec).reshape(1, -1))
    return score


def predict_winner(low, high, circle):
    circle += 1
    if high-low < 2:
        # print(low, high)
        nation1 = id_nation_dict[final16_nation_list[low]]
        nation2 = id_nation_dict[final16_nation_list[high]]
        score = predict_match(score_model, nation1, nation2)
        print(pow(2, circle), (nation1, nation2), score)
        wf.write(str(pow(2, circle))+','+nation1+','+nation2+','+str(score[0])+'\n')
        if score >= 0:
            return low
        else:
            return high
    mid = floor((low+high)/2)
    win1_idx = predict_winner(low, mid, circle)
    win2_idx = predict_winner(mid+1, high, circle)
    nation1 = id_nation_dict[final16_nation_list[win1_idx]]
    nation2 = id_nation_dict[final16_nation_list[win2_idx]]
    score = predict_match(score_model, nation1, nation2)
    print(pow(2, circle), (nation1, nation2), score)
    wf.write(str(pow(2, circle))+','+nation1+','+nation2+','+str(score[0])+'\n')
    if score >= 0:
        return win1_idx
    else:
        return win2_idx


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_log_error
    from sklearn.preprocessing import minmax_scale
    from xgboost import XGBClassifier

    # train
    # load training data
    print('loading training data...')
    history_path = './data/rawdata_elo.txt'
    nation_record_dict = read_history_count.nation_record_count(history_path)
    train_X, train_Y = read_history_count.read_train(history_path, True)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    # train
    print('start training...')

    score_model = XGBClassifier(n_estimators=250, max_depth=7)

    score_model.fit(train_X, train_Y)
    Y_true = minmax_scale(train_Y, feature_range=(0, 1))
    Y_pred = minmax_scale(score_model.predict(train_X), feature_range=(0, 1))
    print('trainset mean log error: %.9f' % mean_squared_log_error(Y_true, Y_pred))

    # predict
    # load prediction data
    print('loading prediction data...')
    sorted_path = './result/promoted_nation.csv'  # promoted teams
    id_nation_dict, four_best3rd = read_id_nation_dict(sorted_path)
    id_path = './data/final16_id_list.txt'  # round16 vs list
    final16_nation_list = read_final16_nation_list(id_path, id_nation_dict, four_best3rd)
    euro2016_path = './data/euro2016.csv'  # euro2016 info
    nation_info_dict, group_nation_dict = read_euro2016info.read_euro2016(euro2016_path)

    # predict
    print('Knockout prediction results:')
    wf = open('./result/knockout_result.csv', 'w')
    champion_id = predict_winner(0, 15, 0)
    print(id_nation_dict[final16_nation_list[int(champion_id)]])
    wf.close()
