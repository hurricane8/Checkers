# -*- coding: utf-8 -*-
from preproc2 import convert_num_to_coord, convert_coord_to_num, parser
import numpy as np





def kill_detector(pole, p):  
    kill_result = []
    fx, fy = convert_num_to_coord(p)
    cell = pole[p]
    if cell == '3':
        v1 = (fx + 2, fy + 2)
        v2 = (fx + 2, fy - 2)
        v3 = (fx - 2, fy + 2)
        v4 = (fx - 2, fy - 2)#воpможные варианты взятия

        for v in (v1, v2, v3, v4):
            if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1: #проверка на выход за границы
                continue
            try:
                if not (pole[convert_coord_to_num(v[0], v[1])] in ('0', '1')): #проверка на свободную клетку
                    continue
            except:
                continue
            if (pole[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)] in ('0', '1', '3', '5')):
                #проверка на занятость клетки между ними
                continue
            kill_result.append((fx, fy, v[0], v[1]))
            return len(kill_result) != 0

    elif cell == '5':
        for i in range(1, 8):  # расст до фигуры
            for j in range(1, 8):  # расст от фигуры до кон клетки
                v1 = (fx + i, fy + i)  # это координаты потенциальной бьющейся фигуры
                v2 = (fx + i, fy - i)
                v3 = (fx - i, fy + i)
                v4 = (fx - i, fy - i)
                u1 = (fx + i + j, fy + i + j)  # где окажемся
                u2 = (fx + i + j, fy - i - j)
                u3 = (fx - i - j, fy + i + j)
                u4 = (fx - i - j, fy - i - j)

                for v, u in zip((v1, v2, v3, v4), (u1, u2, u3, u4)):
                    if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1 or \
                            u[0] > 8 or u[1] > 8 or u[0] < 1 or u[1] < 1:
                        continue

                    if (pole[convert_coord_to_num(v[0], v[1])] in ('2', '4')) \
                            and (pole[convert_coord_to_num(u[0], u[1])] in ('0', '1')):
                        return True

    return len(kill_result) != 0

def reduce_pole(pole):
    result = []
    for i in range(1, 8+1):
        for j in range(1 ,8+1):
            if (i % 2 != 0) and (j % 2 != 0) or (i % 2 == 0) and (j % 2 == 0):
                result.append(pole[convert_coord_to_num(i,j)])
    return result

def generate_all_possible_steps(pole):
    generate_all_possible_steps(pole, 0)

def generate_all_possible_steps(pole, _p):  # принимаем на вход поле и текущую позицию фигуры, которой было совершено на предыдущем ходе
    result = []
    kill_need = False
    victim_coord = 0
    for p, cell in enumerate(pole): #regular steps
        victim_coord = 0
        if _p != -1 and p != _p:
            continue
        if cell in ('2', '4'):
            continue
        fx, fy = convert_num_to_coord(p)
        if (kill_detector(pole, p)):
            kill_need = True
            result = []
            break

        if cell == '5':
            t = [True] * 4
            for i in range(1, 8):
                victim_coord = 0
                v1 = (fx + i, fy + i)
                v2 = (fx + i, fy - i)
                v3 = (fx - i, fy + i)
                v4 = (fx - i, fy - i)
                for k, v in enumerate((v1, v2, v3, v4)):
                    if not t[k]:
                        continue
                    if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1:
                        continue
                    if not (pole[convert_coord_to_num(v[0], v[1])] in ('0', '1')):
                        t[k] = False
                        continue
                    pole_new = pole.copy()
                    pole_new[p] = '1'
                    pole_new[convert_coord_to_num(v[0], v[1])] = '5'
                    result.append(((fx, fy, v[0], v[1]), pole_new, 0, -1))

        elif cell == '3':
            v1 = (fx + 1, fy - 1)
            v2 = (fx - 1, fy - 1)

            for v in (v1, v2):
                victim_coord = 0
                if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1:
                    continue
                if not (pole[convert_coord_to_num(v[0], v[1])] in ('0', '1')):
                    continue
                pole_new = pole.copy()
                pole_new[p] = '1'
                pole_new[convert_coord_to_num(v[0], v[1])] = '3'
                result.append(((fx, fy, v[0], v[1]), pole_new, 0, -1))

    if (kill_need):
        for p, cell in enumerate(pole): #killing steps
            victim_coord = 0
            if _p != -1 and p != _p:
                continue
            fx, fy = convert_num_to_coord(p)
            if cell == '3':
                v1 = (fx + 2, fy + 2)
                v2 = (fx + 2, fy - 2)
                v3 = (fx - 2, fy + 2)
                v4 = (fx - 2, fy - 2)

                for v in (v1, v2, v3, v4):
                    victim_coord = 0
                    if v[0] > 8 or v[1] > 8 or v[1] < 1 or v[1] < 1:
                        continue
                    try:
                        if not (pole[convert_coord_to_num(v[0], v[1])] in ('0', '1')):
                            continue
                    except:
                        continue
                    if (pole[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)]
                            in ('0', '1', '3', '5')):
                        continue
                    pole_new = pole.copy()
                    pole_new[p] = '1'
                    pole_new[convert_coord_to_num(v[0], v[1])] = '3'
                    victim = pole_new[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)]
                    pole_new[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)] = '1'
                    victim_coord = convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)
                    result.append(((fx, fy, v[0], v[1]), pole_new, int(victim), victim_coord))

            elif cell == '5':
                t = [True] * 4
                for i in range(1, 8):  # расст до фигуры
                    for j in range(1, 8):  # расст от фигуры до кон клетки
                        victim_coord = 0

                        v1 = (fx + i, fy + i)  # это координаты потенциальной бьющейся фигуры
                        v2 = (fx + i, fy - i)
                        v3 = (fx - i, fy + i)
                        v4 = (fx - i, fy - i)
                        u1 = (fx + i + j, fy + i + j)  # где окажемся
                        u2 = (fx + i + j, fy - i - j)
                        u3 = (fx - i - j, fy + i + j)
                        u4 = (fx - i - j, fy - i - j)




                        for k, v, u in zip(range(0,4), (v1, v2, v3, v4), (u1, u2, u3, u4)):
                            if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1 or \
                                    u[0] > 8 or u[1] > 8 or u[0] < 1 or u[1] < 1:
                                continue



                            if (pole[convert_coord_to_num(v[0], v[1])] in ('2', '4') and t[k]):
                                if not pole[convert_coord_to_num(u[0], u[1])] in ('0', '1'):
                                    t[k] = False
                                else:
                                    pole_new = pole.copy()
                                    pole_new[p] = '1'
                                    victim = pole_new[convert_coord_to_num(v[0], v[1])]
                                    pole_new[convert_coord_to_num(u[0], u[1])] = '5'
                                    pole_new[convert_coord_to_num(v[0], v[1])] = '1'
                                    victim_coord = convert_coord_to_num(v[0], v[1])
                                    result.append(((fx, fy, u[0], u[1]), pole_new, int(victim), victim_coord))

    return result, kill_need

def convert_X_data(pole, p_step, kill_need):
    step, pole_new, victim, victim_coord = p_step
    pole = [int(x) for x in pole]
    pole_new = [int(x) for x in pole_new]
    possible_X = np.asarray(reduce_pole(pole) + reduce_pole(pole_new)
                            +reduce_pole([int(x1) - int(x2) for x1, x2 in zip(pole, pole_new)])
                            + [pole[convert_coord_to_num(step[0], step[1])]]
                            + list(step) + [1 if kill_need else 0] + [victim] + [victim_coord], dtype='float32')
    return possible_X

## FOR TESTING:
if __name__ == "__main__":
    import pickle

    with open('data.pickle', 'rb') as f:
        games_poles_black, games_winning_steps_black = pickle.load(f)

    X = []
    y = []
    sample_weights = []
    X_dict = {}
    tt = 0
    for poles_black, winning_steps_black in zip(games_poles_black, games_winning_steps_black):
        prev_X = []
        prev_y = []
        good_X = []
        prev_sample_weights = []
        prev_kill = False
        prev_p = -1
        for i, pole, win_step in zip(range(len(poles_black)), poles_black, winning_steps_black):
            if i==0:
                pass
            if convert_coord_to_num(win_step[0], win_step[1]) != prev_p:
                prev_p = -1

            possible_steps = generate_all_possible_steps(pole, prev_p)
            good_X = []
            flag = False
            for p_step in possible_steps[0]:
                step, _, _, _ = p_step
                if flag:
                    pass
                  #  break

                possible_X = convert_X_data(pole, p_step, possible_steps[1])
               # possible_X = np.asarray(np.concatenate((np.concatenate(keras.utils.to_categorical([int(x) for x in pole[::]],6)),
                #                        np.concatenate(keras.utils.to_categorical(list(step),9))) ), dtype='float32') #+ [1 if possible_steps[1] else 0]
                #patch_X = np.zeros((46 - len(prev_X) - 1, possible_X.shape[0]))
                #patch_y = np.zeros((46 - len(prev_X) - 1, 2))
                key = tuple(possible_X)
                if step == win_step:

                    if not key in X_dict.keys():
                        X_dict[key] = 1
                    else:
                        print(X_dict[key],(X_dict[key]*0.5 +  1*0.5))
                        if X_dict[key] < 0.5:
                            print('wwwwwwwwww', X_dict[key])
                        X_dict[key] = (X_dict[key]*0.5 +  1*0.5)
                        tt += 1
                    flag = True
                    new_prev_p = convert_coord_to_num(step[2], step[3]) if possible_steps[1] else -1
                    possible_y = 1# - np.random.rand(1)[0]#np.asarray([1, 0], dtype='float32')
                    good_X = possible_X
                    sw = 1

                else:
                    if not key in X_dict.keys():
                        X_dict[key] = 0
                    else:
                        dividor = 2
                        if (X_dict[key]>0.5):

                            X_dict[key] = X_dict[key] / 2
                        else:
                            print(X_dict[key],X_dict[key]/dividor)
                            X_dict[key] = X_dict[key]/dividor
                    possible_y = 0 #+ np.random.rand(1)[0]#np.asarray([0, 1], dtype='float32')
                    sw = 1

                if prev_X != []:
                    X.append(possible_X)
                    y.append(possible_y)
                    #X.append(np.concatenate((np.asarray(prev_X),
                                            # np.asarray([possible_X]), patch_X), axis=0))
                   # y.append(np.concatenate((prev_y, np.asarray([possible_y]), patch_y), axis=0))
                else:
                    X.append(possible_X)
                    y.append(possible_y)
                   # X.append(np.concatenate((np.asarray([possible_X]), patch_X), axis=0))
                   # y.append(np.concatenate((np.asarray([possible_y]), patch_y), axis=0))

                sample_weights.append(sw)

            if good_X == []:
                print(win_step, possible_steps, pole[convert_coord_to_num(win_step[0], win_step[1])])
            prev_p = new_prev_p
            prev_X.append(good_X)
            prev_y.append(np.asarray([0], dtype='float32'))
            prev_sample_weights.append(sw)
    print(X[15])

    n1 = 0
    n2 = 0
    for _y in y:
        if _y == 1:
            n1 += 1
        else:
            n2 += 1
    print(n1, n2, n1/(n1+n2), n2/(n1+n2))

    print('Data collecting has been finished')



    from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor


    cl = RandomForestRegressor(200, max_features=None, max_depth=None, criterion='mse', n_jobs=-1)
  #  cl = GradientBoostingRegressor(n_estimators=250,max_depth=None)
    #, class_weight='balanced')
    from sklearn.model_selection import train_test_split
    X, y = zip(*X_dict.items())
    X, Xtest, y, ytest = train_test_split(X, y, test_size=0.20)
    n1 = 0
    n2 = 0
    for _y in y:
        if _y == 0.5:
            n1 += 1

    print(X[0])
    print(n1, len(X))

    cl.fit(X,y)
    cl.score(X,y)
    print(cl.feature_importances_)
    print(np.argsort(cl.feature_importances_))
    print(cl.score(Xtest,ytest))
    with open('classifier.pickle', 'wb') as file:
        pickle.dump(cl, file=file, protocol=pickle.HIGHEST_PROTOCOL)
'''
    import keras



    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer((None, 420)))#2*32+4+1)))
    keras.layers.Masking(mask_value=0.0)
   # model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Conv1D(100, kernel_size=3))
 #   model.add(keras.layers.Flatten())
  #  model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
    #model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
    #model.add(keras.layers.LSTM(10, activation=keras.activations.elu, return_sequences=True, use_bias=True))
    #model.add(keras.layers.LSTM(10, activation=keras.activations.elu, return_sequences=True, use_bias=True))
    #model.add(keras.layers.LSTM(10, activation=keras.activations.elu, return_sequences=True, use_bias=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(2, activation='softmax', use_bias=True)))
    #optimizer = keras.optimizers.sgd(lr=0.001, momentum=0.8, nesterov=True)
    optimizer = keras.optimizers.Adagrad()

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])#, sample_weight_mode="temporal")
    print(len(X))



    from keras.callbacks import EarlyStopping
    early = EarlyStopping('val_loss', patience=100, restore_best_weights=True, verbose=1)
    model.fit(np.asarray(X), np.asarray(y), validation_split=0.1, epochs=300, batch_size=800, initial_epoch=0,
              class_weight=[0.7,0.3], callbacks=[early], verbose=2)
   # for ep, _X, _y, _w in zip(range(2000), X, y, sample_weights): #list(range(len(X)))
      #  model.fit(np.asarray([_X]), np.asarray([_y]), validation_split=0., epochs=1, batch_size=1, \
                 # initial_epoch=0, sample_weight=np.asarray([_w], dtype='float32'))#, class_weight={0 : n2/(n1+n2), 1: n1/(n1+n2)})


    for  _X, _y in zip( X, y):
        arg = np.argwhere((_X!=0).max(axis=1))[-1][0]
        print(model.predict_classes(np.asarray([_X]))[0][arg])

'''
