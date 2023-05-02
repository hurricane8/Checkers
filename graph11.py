import cocos
from cocos.layer.base_layers import Layer
from cocos.actions.interval_actions import MoveTo, MoveBy, ScaleBy, RotateBy ,Repeat, Delay, FadeOut
from random import randint
from cocos.actions.instant_actions import CallFunc, Place
from cocos.sprite import Sprite
from math import sqrt
import pyglet.window
from pyglet.event import EventDispatcher
from preproc2 import convert_coord_to_num, convert_num_to_coord, rev_pole, rev_steps
size = 3/4 #во сколько раз увеличиваем
import sklearn
import pickle
import time
from AI_engine1 import generate_all_possible_steps, convert_X_data, kill_detector
from multiprocessing import Process
import sys
if __name__ == '__main__':
    with open('classifier.pickle', 'rb') as file:
        cl = pickle.load(file)


class Player(Layer):
    is_event_handler = True

    def on_mouse_press (self, x, y, buttons, modifiers):
        if len(self.get_ancestor(Level2).children[0][1].children) > 1:
            print(self.get_ancestor(Level2).children[0][1].children)
            sprite = self.get_ancestor(Level2).children[0][1].children[1][1]
            if (abs(sprite.x - x)<=sprite.width/2 and abs(sprite.y - y)<=sprite.height/2):
               # cocos.director.director.unscaled_resize_window(int(1000*size), int(400*size))
                new_scene = Level1()

                #cocos.director.director.window.set_size(int(1000 * size), int(500 * size))
                cocos.director.director.replace(new_scene)
                sprite.kill()
                
                sprite1 = self.get_ancestor(Level2).children[0][1].children[1][1]
                sprite1.kill()
             

        if len(self.get_ancestor(Level2).children[0][1].children) > 1:
            print(self.get_ancestor(Level2).children[0][1].children)
            sprite = self.get_ancestor(Level2).children[0][1].children[2][1]
            if (abs(sprite.x - x)<=sprite.width/2 and abs(sprite.y - y)<=sprite.height/2):
                #sys.exit()
                sprite.kill()
                exit()
                self.get_ancestor(Level2).children[0][1].children[1][1].kill()
       # cocos.director.director.run(Level1())


class Level2(cocos.scene.Scene):
   def __init__(self):
        super().__init__()
        #cocos.director.director.init( width=int(800*size), height=int(800*size), caption="fon")
        scene = cocos.scene.Scene()
        
        Background = cocos.layer.base_layers.Layer()
        fon = cocos.sprite.Sprite('images/fon2.jpg', position=(500*size, 400*size),scale=size)
        self.add(Background)
        Background.add(fon)

        l11 = Player()
        sp192 = cocos.sprite.Sprite('images/knopka.png', position=(825*size, 600*size), scale = 2/3*size) #начало игры
        Background.add(sp192)
        self.add(l11)

        game_end = cocos.sprite.Sprite('images/game-end.png', position=(825*size,450*size),scale = 2/3*size)
        Background.add(game_end)

        


        

class PlayerInput(Layer, EventDispatcher):


    is_event_handler = True

    is_players_step = True

    AI_steps_queue = []

    _sprite = None
    _x = 0
    _y = 0

    pole = []
    _player_kill_need = False
    _player_possible_steps = []

    def sprite_coords_to_pole_coords(self, x, y):
        pole_x = x / size // 100 + 1
        pole_y = y / size // 100 + 1

        return pole_x, pole_y


    def refresh_pole(self):
        self.pole = []
        for i in range(4):
            for j in range(4):
                self.pole.append('0')
                self.pole.append('1')
            for j in range(4):
                self.pole.append('1')
                self.pole.append('0')
       # print(pole)
        for _, sprite in self.get_ancestor(Level1).children[1][1].children: #white
            index = convert_coord_to_num((sprite.x)/size//100 + 1, (sprite.y)/size//100 + 1)
            if convert_num_to_coord(index)[1] == 8:
                wd = cocos.sprite.Sprite('images/bd.png', position=(sprite.x, sprite.y), scale=size)
                wd.add(cocos.sprite.Sprite('images/bd.png'))
                self.get_ancestor(Level1).children[1][1].add(wd)
                sprite.kill()
                self.pole[index] = '4'
            else:
                if len(sprite.children) > 0:
                    self.pole[index] = '4'
                else:
                    self.pole[index] = '2'

        for _, sprite in self.get_ancestor(Level1).children[2][1].children: #black
            index = convert_coord_to_num((sprite.x)/size//100 + 1, (sprite.y)/size//100 +1)
            if convert_num_to_coord(index)[1] == 1:
                bd = cocos.sprite.Sprite('images/kd.png', position=(sprite.x, sprite.y), scale=size)
                bd.add(cocos.sprite.Sprite('images/kd.png'))
                self.get_ancestor(Level1).children[2][1].add(bd)
                sprite.kill()
                self.pole[index] = '5'
            else:
                if len(sprite.children) > 0:
                    self.pole[index] = '5'
                else:
                    self.pole[index] = '3'
     #   print(pole)




    def on_mouse_press(self, x, y, buttons, modifiers):
        if not self.is_players_step:
            return

        if abs(self.get_ancestor(Level1).children[5][1].x - x)<125/3*size and abs(self.get_ancestor(Level1).children[5][1].y - y)<41/3*size:
            sys.exit()
            
        for _, sprite in self.get_ancestor(Level1).children[1][1].children:
            if (abs(sprite.x - x) <= sprite.width / 2 and abs(sprite.y - y) <= sprite.height / 2):
                self._sprite = sprite
                self._x = int(sprite.x)
                self._y = int(sprite.y)
                break



    def on_mouse_drag (self, x, y, dx, dy, buttons, modifiers):
        if not self.is_players_step:
            return
            #sp = self.get_ancestor(Level1).children[-1][1].children[0][1]
        if self._sprite != None:
            self._sprite.x += dx
            self._sprite.y += dy

    def on_mouse_release (self, x, y,  buttons, modifiers):
        if not self.is_players_step:
            return

        a=1
        killed = False
        new_x = int(x) // (100 * size) * 100 * size + 50 * size
        new_y = int(y) // (100 * size) * 100 * size + 50 * size
        fx, fy = self.sprite_coords_to_pole_coords(self._x, self._y)
        lx, ly = self.sprite_coords_to_pole_coords(new_x, new_y)
        step = (fx, fy, lx, ly)

        #sp = self.get_ancestor(Level1).children[-1][1].children[0][1]
        if self._sprite != None:
            if not (step in self._player_possible_steps):
                self._sprite.x = int(self._x)
                self._sprite.y = int(self._y)
                self._sprite = None
                return

            if not self._player_kill_need:
                killed = False
                self._sprite.x = new_x
                self._sprite.y = new_y
                self._sprite = None
                self.dispatch_event('on_Players_step_finished', killed, -1)
                return
            else:
                killed = True
                self._sprite.x = new_x
                self._sprite.y = new_y
                direction_x = 1 if (lx > fx) else -1
                direction_y = 1 if (ly > fy) else -1
                for i in range(1,8):
                    victim_x = fx + i * direction_x
                    victim_y = fy + i * direction_y
                    if self.pole[convert_coord_to_num(victim_x,victim_y)] in ('3', '5'):
                        for _, sprite in self.get_ancestor(Level1).children[2][1].children:
                            if (victim_x, victim_y) == self.sprite_coords_to_pole_coords(sprite.x,sprite.y):
                                sprite.kill()
                                self.dispatch_event('on_Players_step_finished', killed, convert_coord_to_num(lx,ly))
                                self._sprite = None
                                return
        self._sprite = None








    def on_Players_step_finished(self, killed, players_prev_p):
        self.refresh_pole()
        self.Players_step_precomputing(players_prev_p)
        if killed and len(self._player_possible_steps) != 0 and self._player_kill_need:
            pass
        else:
            self.is_players_step = False
            self.AI_steps_queue = []
            self.AI_step()

    def Players_step_precomputing(self, players_prev_p):
        reversed_pole = rev_pole(self.pole)

        possible_steps, kill_need = generate_all_possible_steps(reversed_pole, -1 if players_prev_p == -1 else 63-players_prev_p)

        if len(possible_steps) > 0:
            stepsss, _, _, _ = zip(*possible_steps)
        elif players_prev_p == -1:
            n_white = len(self.get_ancestor(Level1).children[1][1].children)
            n_black = len(self.get_ancestor(Level1).children[2][1].children)
            l3 = Level3(n_white=n_white,n_black=n_black, prev_player='black')
            cocos.director.director.replace(l3)
            return

        new_steps = []
        if len(possible_steps) > 0:
            steps, _, _, _ = zip(*possible_steps)
            for step in steps:
                step = [step[0], step[1], '_', step[2],step[3]]
                step = rev_steps(step)
                new_steps.append((int(step[0]), int(step[1]), int(step[3]), int(step[4])))
        steps = new_steps
        print(steps)
        self._player_possible_steps = steps
        self._player_kill_need = kill_need

    def on_AI_step_finished(self, killed, prev_p):
        #  self.refresh_pole()
        print('\n\n\n\n')
        if killed and prev_p != -2:
            self.AI_step(prev_p)
        else:
            print(self.AI_steps_queue)
            result = MoveBy((0, 0), 0.001)
            if len(self.AI_steps_queue) != 0:
                _, victim_coords, _ = zip(*self.AI_steps_queue)
                for p_step, victim_coord, kill_need in self.AI_steps_queue:
                    v1 = p_step[0:2]
                    u1 = p_step[2:4]

                    result += MoveTo((int(int(u1[0]) * 100 * size - 50 * size), int(int(u1[1]) * 100 * size - 50 * size)),
                                     1)
                print(result)

                self.Move_Black((self.AI_steps_queue[0][0][0], self.AI_steps_queue[0][0][1], p_step[2], p_step[3]),
                                victim_coords, kill_need, result)
                self.Players_step_precomputing(-1)
            self.is_players_step = True






    def Move_Black(self, step, victim_coords, kill_need, acts):
        v1 = step[0:2]
        u1 = step[2:4]


        print(step[-1])


        for _, sprite in self.get_ancestor(Level1).children[2][1].children:
            if (abs(sprite.x - (int(v1[0]) * 100 * size - 50 * size)) <= sprite.width / 2 and abs(
                    sprite.y - (int(v1[1]) * 100 * size - 50 * size)) <= sprite.height / 2):

                sprite.do(acts)

        if kill_need :
            for victim_coord in victim_coords:
                victim_coord_x, victim_coord_y = convert_num_to_coord(victim_coord)
                print(victim_coord_x, victim_coord_y)
                for _, sprite in self.get_ancestor(Level1).children[1][1].children:
                    g_coord_x = int(victim_coord_x) * 100 * size - 50 * size
                    g_coord_y = int(victim_coord_y) * 100 * size - 50 * size
                    if (abs(sprite.x - g_coord_x) <= sprite.width / 2 and abs(
                            sprite.y - g_coord_y) <= sprite.height / 2):
                        t_sprite = cocos.sprite.Sprite('images/b.png', position=(g_coord_x, g_coord_y),scale=size)
                        self.get_ancestor(Level1).children[3][1].add(t_sprite)
                        t_sprite.do(FadeOut(1))
                        sprite.kill()




    def AI_step(self, prev_p=-1):
        global t
        possible_steps = generate_all_possible_steps(self.pole, prev_p)

        p_steps = possible_steps[0]  # сами ходы
        kill_need = possible_steps[1]  # необходимость взятия

        if len(p_steps) == 0 and prev_p == -1:
            print(possible_steps)
            n_white = len(self.get_ancestor(Level1).children[1][1].children)
            n_black = len(self.get_ancestor(Level1).children[2][1].children)
            l3 = Level3(n_white=n_white,n_black=n_black, prev_player='white')
            cocos.director.director.replace(l3)
            return




        if len(p_steps) != 0 and (prev_p == -1 or kill_need):
            print(self.pole)
            print(prev_p)
            print(possible_steps)
            best_step = p_steps[0]
            best_prob = 0

            print(len(p_steps))
            for p_step in p_steps:
                X = convert_X_data(self.pole, p_step, kill_need)
                prob = cl.predict([X])[0]
                print(prob)
                if prob > best_prob:
                    best_prob = prob
                    best_step = p_step

            p_step = best_step[0]
            pole_new = best_step[1]
            victim_coord = best_step[-1]
            prev_p = convert_coord_to_num(p_step[2], p_step[3])
            print(p_step, best_prob)
            print(kill_need)
            self.pole = pole_new
            self.AI_steps_queue.append((p_step, victim_coord, kill_need))

           # time.sleep(0.8)
            self.dispatch_event('on_AI_step_finished', kill_need, prev_p)


        else:
            self.dispatch_event('on_AI_step_finished', kill_need, -2)






PlayerInput.register_event_type('on_Players_step_finished')
PlayerInput.register_event_type('on_AI_step_finished')

class Player_end(Layer):
    is_event_handler = True

    def on_mouse_press (self, x, y, buttons, modifiers):
        if len(self.get_ancestor(Level3).children) > 1:
            sprite = self.get_ancestor(Level3).children[2][1]
            if (abs(sprite.x - x)<=sprite.width/2 and abs(sprite.y - y)<=sprite.height/2):
               # cocos.director.director.unscaled_resize_window(int(1000*size), int(400*size))
                new_scene = Level2()

                #cocos.director.director.window.set_size(int(1000 * size), int(500 * size))
                cocos.director.director.replace(new_scene)
                sprite.kill()
                
                sprite1 = self.get_ancestor(Level3).children[1][1]
                sprite1.kill()
             

        if len(self.get_ancestor(Level3).children[0][1].children) > 1:
            sprite = self.get_ancestor(Level3).children[1][1]
            if (abs(sprite.x - x)<=sprite.width/2 and abs(sprite.y - y)<=sprite.height/2):
                sys.exit()
                sprite.kill()
                self.get_ancestor(Level3).children[1][1].kill()


class Level3(cocos.scene.Scene):
    def __init__(self, n_white, n_black, prev_player):
        global size
        if n_white < n_black:
            winner = 'black'
        elif n_white > n_black:
            winner = 'white'
        elif n_white == n_black:
            winner = prev_player
        super().__init__()
        Background = cocos.layer.base_layers.Layer()

        fon = cocos.sprite.Sprite('images/bg_konez.jpg', position=(500 * size, 400 * size), scale=size)
        Background.add(fon)
        self.add(Background)
        
        game_end = cocos.sprite.Sprite('images/game-end.png', position=(900 * size, 600 * size), scale=1 / 3 * size)
        self.add(game_end)

        knopka_v_menu = cocos.sprite.Sprite('images/knopka_v_menu.png', position=(900 * size, 300 * size), scale=1 / 3 * size)
        self.add(knopka_v_menu)
        
        if (winner == 'white'):
            win = cocos.sprite.Sprite('images/win.png', position=(500 * size, 400 * size), scale = size)
            self.add(win)
            
        if (winner == 'black'):
            lose = cocos.sprite.Sprite('images/lose.png', position=(500 * size, 400 * size), scale = size)
            self.add(lose)

        Pl = Player_end()
        self.add(Player_end())

        

        


class Level1(cocos.scene.Scene):
    def __init__(self):
        global size
        super().__init__()
        #self.
        #cocos.director.director.init(width=int(1000*size), height=int(800*size), caption="fon")

        Background = cocos.layer.base_layers.Layer()
        fon = cocos.sprite.Sprite('images/background.jpg', position=(500*size,400*size),scale=size)
        self.add(Background)
        Background.add(fon)



        White = cocos.layer.base_layers.Layer()
        b1 = cocos.sprite.Sprite('images/b.png', position=(50*size,50*size),scale=size)

        b2 = cocos.sprite.Sprite('images/b.png', position=(250*size,50*size),scale=size)
        b3 = cocos.sprite.Sprite('images/b.png', position=(450*size,50*size),scale=size)
        b4 = cocos.sprite.Sprite('images/b.png', position=(650*size,50*size),scale=size)
        b5 = cocos.sprite.Sprite('images/b.png', position=(150*size,150*size),scale=size)
        b6 = cocos.sprite.Sprite('images/b.png', position=(350*size,150*size),scale=size)
        b7 = cocos.sprite.Sprite('images/b.png', position=(550*size,150*size),scale=size)
        b8 = cocos.sprite.Sprite('images/b.png', position=(750*size,150*size),scale=size)
        b9 = cocos.sprite.Sprite('images/b.png', position=(50*size,250*size),scale=size)
        b10 = cocos.sprite.Sprite('images/b.png', position=(250*size,250*size),scale=size)
        b11 = cocos.sprite.Sprite('images/b.png', position=(450*size,250*size),scale=size)
        b12 = cocos.sprite.Sprite('images/b.png', position=(650*size,250*size),scale=size)
        self.add(White)
        White.add(b1)
        White.add(b2)
        White.add(b3)
        White.add(b4)
        White.add(b5)
        White.add(b6)
        White.add(b7)
        White.add(b8)
        White.add(b9)
        White.add(b10)
        White.add(b11)
        White.add(b12)


        Black = cocos.layer.base_layers.Layer()
        k1 = cocos.sprite.Sprite('images/k.png', position=(150*size,750*size),scale=size)
        k2 = cocos.sprite.Sprite('images/k.png', position=(350*size,750*size),scale=size)
        k3 = cocos.sprite.Sprite('images/k.png', position=(550*size,750*size),scale=size)
        k4 = cocos.sprite.Sprite('images/k.png', position=(750*size,750*size),scale=size)
        k5 = cocos.sprite.Sprite('images/k.png', position=(50*size,650*size),scale=size)
        k6 = cocos.sprite.Sprite('images/k.png', position=(250*size,650*size),scale=size)
        k7 = cocos.sprite.Sprite('images/k.png', position=(450*size,650*size),scale=size)
        k8 = cocos.sprite.Sprite('images/k.png', position=(650*size,650*size),scale=size)
        k9 = cocos.sprite.Sprite('images/k.png', position=(150*size,550*size),scale=size)
        k10 = cocos.sprite.Sprite('images/k.png', position=(350*size,550*size),scale=size)
        k11 = cocos.sprite.Sprite('images/k.png', position=(550*size,550*size),scale=size)
        k12 = cocos.sprite.Sprite('images/k.png', position=(750*size,550*size),scale=size)
        self.add(Black)
        Black.add(k1)
        Black.add(k2)
        Black.add(k3)
        Black.add(k4)
        Black.add(k5)
        Black.add(k6)
        Black.add(k7)
        Black.add(k8)
        Black.add(k9)
        Black.add(k10)
        Black.add(k11)
        Black.add(k12)

        Died = cocos.layer.base_layers.Layer()
        self.add(Died)

        l = PlayerInput()
        self.add(l)


        game_end = cocos.sprite.Sprite('images/game-end.png', position=(900*size,600*size),scale=1/3*size)
        self.add(game_end)


        l.refresh_pole()
        l.Players_step_precomputing(-1)
       # self.add(Gamer((1,6), (2,5)))



if __name__ == '__main__':

    cocos.director.director.init(width=int(1000*size), height=int(800*size))
    l1 = Level2()
    cocos.director.director.run(l1)




