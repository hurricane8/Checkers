# -*- coding: utf-8 -*-
import urllib.request


def rev_pole(pole): #зеркалим доску
    _pole = pole.copy()
    _pole = ['6' if x=='2' else x for x in _pole]
    _pole = ['2' if x=='3' else x for x in _pole]
    _pole = ['3' if x=='6' else x for x in _pole]

    _pole = ['6' if x=='4' else x for x in _pole]
    _pole = ['4' if x=='5' else x for x in _pole]
    _pole = ['5' if x=='6' else x for x in _pole]
    _pole = list(reversed(_pole))
    return _pole

def rev_steps(step): #зеркалим координаты полей
    fx, fy, divider, lx, ly = step
    newfx, newfy = convert_num_to_coord(63 - convert_coord_to_num(fx,fy))
    newlx, newly = convert_num_to_coord(63 - convert_coord_to_num(lx,ly))
    return str(newfx)+str(newfy)+divider+str(newlx)+str(newly)

def letter_to_num(a): #переводим все координаты в цифры
    a = a.replace('a', '1')
    a = a.replace('b', '2')
    a = a.replace('c', '3')
    a = a.replace('d', '4')
    a = a.replace('e', '5')
    a = a.replace('f', '6')
    a = a.replace('g','7')
    a = a.replace('h','8')
    return a


def convert_coord_to_num(x, y): 
    return (8-int(y))*8+(int(x)-1)

def convert_num_to_coord(p): #test this func
    x = p % 8 + 1
    y = 8 - (p // 8)
    return x, y



def parser(tourname, gamenum):
    f = urllib.request.urlopen('http://shashki.org/games/' + tourname + '.files/game' + str(gamenum) + '.htm') #загружаем с сайта архив состояний

    source = f.read().decode("windows-1251").split('\n') #проводим декодировку

    start_index = 0
    for i, v in enumerate(source):  #находим первое вхождение состояний доски
        if 'm[0]=' in v:
            start_index = i
            break

    last_index = start_index
    while ('m[' in source[last_index]):
        last_index += 1

    last_index -= 1 #находим последнее вхождение состояний доски

    poles = [list(v.replace('6', '').split('"')[1]) for v in source[start_index:last_index+1]]  #приводим состояние доски в 64 цифры(64 клетки)


    start_index = 0
    for i, v in enumerate(source):
        if '<a href="javascript:to(' in v: 
            start_index = i  #находим первое вхождение результата партии
            break

    last_index = start_index
    while ('<a href="javascript:to(' in source[last_index]):
        last_index += 1  #находим последнее вхождение результата партии

    last_index -= 1


    steps = [v.replace('/', '').split('<b>')[1] for v in source[start_index:last_index+1]]
    steps = [letter_to_num(v.split('.')[-1]) for v in steps]

    result = source[last_index+1].split('-')
    result[0] = result[0][-1]
    result[1] = result[1][0]
    assert len(poles) == (len(steps) + 1) #проверяем, что состояний доски на 1 больше, чем ходов(тк есть начальное)
    return poles, steps, result


def deep_search(fx, fy, lx, ly, polef, polel, steps, poles, flag):
    if polef == polel:
        return poles, steps, True

    if polef[convert_coord_to_num(fx, fy)] == '5':
        for i in range(1, 8): # расст до фигуры
            for j in range(1, 8): # расст от фигуры до кон клетки
                v1 = (fx + i, fy + i) # это координаты потенциальной бьющейся фигуры
                v2 = (fx + i, fy - i)
                v3 = (fx - i, fy + i)
                v4 = (fx - i, fy - i)
                u1 = (fx + i + j, fy + i + j) # где окажемся
                u2 = (fx + i + j, fy - i - j)
                u3 = (fx - i - j, fy + i + j)
                u4 = (fx - i - j, fy - i - j)

                for v, u in zip((v1, v2, v3, v4), (u1, u2, u3, u4)):
                    if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1 or \
                            u[0] > 8 or u[1] > 8 or u[0] < 1 or u[1] < 1:
                        continue
                    if (pole[convert_coord_to_num(v[0], v[1])] in ('2', '4')) \
                            and (pole[convert_coord_to_num(u[0], u[1])] in ('0', '1')):

                        pole_new = polef.copy()
                        pole_new[convert_coord_to_num(v[0], v[1])] = '1' # сбитая фигура
                        pole_new[convert_coord_to_num(u[0], u[1])] = polef[convert_coord_to_num(fx, fy)] # конечная позиция
                        pole_new[convert_coord_to_num(fx, fy)] = '1'  # поле, с которых пошли теперь пустое
                        steps.append((fx, fy, u[0], u[1]))
                        poles.append(polef)
                        poles_, steps_, flag = deep_search(u[0], u[1], lx, ly, pole_new, polel, steps, poles, flag)
                        if flag:
                            return poles_, steps_, flag


    elif polef[convert_coord_to_num(fx, fy)] == '3': #не дамка
        v1 = (fx + 2, fy + 2) # где окажемся
        v2 = (fx + 2, fy - 2)
        v3 = (fx - 2, fy + 2)
        v4 = (fx - 2, fy - 2)

        for v in (v1, v2, v3, v4):
            if v[0] > 8 or v[1] > 8 or v[0] < 1 or v[1] < 1:
                continue

            try:
                if not (polef[convert_coord_to_num(v[0], v[1])] in ('0', '1')):
                    continue
            except:
                continue

            if (polef[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)] in ('0', '1', '3', '5')):
                continue

            pole_new = polef.copy()
            pole_new[convert_coord_to_num((fx + v[0]) / 2, (fy + v[1]) / 2)] = '1'
            pole_new[convert_coord_to_num(v[0], v[1])] = polef[convert_coord_to_num(fx, fy)]
            pole_new[convert_coord_to_num(fx, fy)] = '1'  # поле, с которых пошли теперь пустое
            steps.append((fx, fy, v[0], v[1]))
            poles.append(polef)
            poles_, steps_, flag = deep_search(v[0], v[1], lx, ly, pole_new, polel, steps, poles, flag)
            if flag:
                return poles_, steps_, flag

    return poles, steps, False



if __name__ == "__main__":
    games = []
    for tournum in range(1, 10):
        for gamenum in range(1, 20):
            try:
                 games.append(parser('km2010/km_2012_tur' + str(tournum), gamenum))
            except:
                pass
    for tourname in ('km2010/km_2011', 'km2010/km_2011chel', 'km2010/km_2010',
                     'kr2009/pg_2009', 'kr2009/kubok_2009','kr2009/kr2009' ):
        for gamenum in range(1, 20):
            try:
                games.append(parser(tourname, gamenum))
            except:
                pass
   # print(games)
#


    games_poles = []
    games_steps = []

    for game in games:
        poles_black_game = []
        winning_steps_black_game = []
        poles = game[0]
        steps = game[1]
        res = game[2]
        if res[1] == '0':
            poles = [rev_pole(pole) for pole in poles]
            steps = [rev_steps(step) for step in steps]

        for i, pole, step in zip(range(0, len(poles)), poles, steps):

            fx, fy, _, lx, ly = step
            if not (pole[convert_coord_to_num(fx, fy)] in ('3','5')):
                continue

            if '-' in step: #обычный ход
                #print(step)
                poles_black_game.append(pole)
                winning_steps_black_game.append((int(fx), int(fy), int(lx), int(ly)))

            if ':' in step: #взятие
                sqrdist = (int(fx) - int(lx))**2 + (int(fy) - int(ly))**2
               # print(sqrdist)
                if sqrdist > 8:#двойное взятие или больше
                    try:
                        dpoles, dsteps, _= deep_search(int(fx), int(fy), int(lx), int(ly), pole, poles[i+1], [], [], False)
                        poles_black_game += dpoles
                        winning_steps_black_game += dsteps


                    except:
                        print(step, pole, poles[i+1])
                else:
                    poles_black_game.append(pole)
                   # print(poles_black)
                    winning_steps_black_game.append((int(fx), int(fy), int(lx), int(ly)))

        games_poles.append(poles_black_game)
        games_steps.append(winning_steps_black_game)

    import pickle  #сохранение базы игр в документ

    file = open('data.pickle', 'wb')

    pickle.dump((games_poles, games_steps), file=file, protocol=pickle.HIGHEST_PROTOCOL)

    file.close()


