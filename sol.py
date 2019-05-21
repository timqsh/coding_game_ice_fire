
import sys
import math
from collections import namedtuple, deque
from enum import Enum
from random import randint, choice, shuffle, seed
from dataclasses import dataclass
from time import time

seed(1337)
start_time = time()

def recruitment_cost(level):
    return [10, 20, 30][level-1]

def upkeep_cost(level):
    return [1, 4, 20][level-1]

class Side(Enum):
    ME = 0
    THEM = 1

class BuildingType(Enum):
    HQ = 0
    MINE = 1
    TOWER = 2

Point = namedtuple("Point", ["x", "y"])
Unit = namedtuple("Unit", ["owner", "id", "level", "x", "y"])
Building = namedtuple("Building", ["owner", "type", "x", "y"])

@dataclass
class Wealth:
    gold: int
    income: int
    opponent_gold: int
    opponent_income: int

@dataclass
class G:
    my_units_pos: set = None
    map: list = None
    mine_spots: list = None
    border_squares: list = None
    available_squares: list = None
    my_units: list = None
g = G()

def log(x):
    print(x, file=sys.stderr)

def initial_input():
    number_mine_spots = int(input())
    mine_spots = []
    for i in range(number_mine_spots):
        x, y = [int(j) for j in input().split()]
        mine_spots.append(Point(x, y))
    return mine_spots

def turn_input():
    wealth = Wealth(*(int(input()) for i in range(4)))
    
    gamemap = []
    for i in range(12):
        line = list(input())
        gamemap.append(line)

    building_count = int(input())
    buildings = []
    for i in range(building_count):
        params = [int(j) for j in input().split()]
        params[0] = Side(params[0])
        params[1] = BuildingType(params[1])
        buildings.append(Building(*params))

    unit_count = int(input())
    units = []
    for i in range(unit_count):
        params = [int(j) for j in input().split()]
        params[0] = Side(params[0])
        units.append(Unit(*params))

    return wealth, gamemap, buildings, units

def neighbors(point, randomize=False):
    res = []
    if point.x < 11:
        res.append(Point(point.x+1, point.y))
    if point.x > 0:
        res.append(Point(point.x-1, point.y))
    if point.y < 11:
        res.append(Point(point.x, point.y+1))
    if point.y > 0:
        res.append(Point(point.x, point.y-1))

    res = [r for r in res if g.map[r.y][r.x] != "#"]

    if randomize:
        shuffle(res)
        
    res = [r for r in res if Point(r.x, r.y) not in g.my_units_pos]

    return res

def bfs(start, end):

    def next_move(prev):
        cur = end
        while prev[prev[cur]]:
            cur = prev[cur]
        return cur

    q = deque()
    q.append(start)
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            return next_move(prev)
        for pos in neighbors(cur, randomize=True):
            if pos not in prev:
                prev[pos] = cur
                q.append(pos)
        
        # i = 0
        # delta_time = time() - start_time
        # if delta_time > 0.2:
        #     if i%100==0:
        #         log(delta_time)
        #     i+=1

def occupied(point, gamemap, *collections):
    for collection in collections:
        for item in collection:
            if point.x == item.x and point.y == item.y:
                return True
    if gamemap[point.y][point.x] == "#":
        return True
    return False

def calculate_globals(gamemap, buildings, units):
    g.my_units_pos={Point(u.x, u.y) for u in units if u.owner == Side.ME}
    g.map = gamemap
  
    g.available_squares = set()
    my_squares = set()
    g.my_units=[u for u in units if u.owner == Side.ME]
    for x in range(12):
        for y in range(12):
            if g.map[y][x] == "O":
                for n in neighbors(Point(x, y)):
                    if not occupied(n, g.map, g.my_units, buildings):
                        g.available_squares.add(n)
                if not occupied(Point(x, y), g.map, g.my_units, buildings):
                    g.available_squares.add(Point(x, y))
                    my_squares.add(Point(x, y))
    g.border_squares = list(g.available_squares - my_squares)

def kill_by_spawn(enemy_level, my_level, units, wealth, commands):
    enemy_units = [u for u in units if u.owner == Side.THEM and u.level == enemy_level]
    enemy_positions = {Point(u.x, u.y) for u in enemy_units}
    available_enemies = [s for s in g.border_squares if s in enemy_positions]
    while wealth.gold >= recruitment_cost(my_level) and wealth.income >= 0 and available_enemies:
        spawn_point = choice(available_enemies)
        commands.append(f"TRAIN {my_level} {spawn_point.x} {spawn_point.y}")
        wealth.gold -= recruitment_cost(my_level)
        wealth.income -= upkeep_cost(my_level)
        available_enemies.remove(spawn_point)

def spawn_level_1_on_border(wealth, commands):
    while wealth.gold >= 10 and wealth.income >= 0 and g.border_squares:
        spawn_point = choice(g.border_squares)
        commands.append(f"TRAIN 1 {spawn_point.x} {spawn_point.y}")
        wealth.gold -= 10
        wealth.income -= 1
        g.border_squares.remove(spawn_point)

def make_move(wealth, gamemap, buildings, units):
    commands=[]

    calculate_globals(gamemap, buildings, units)
    
    # TRAIN
    kill_by_spawn(3, 3, units, wealth, commands)
    kill_by_spawn(2, 3, units, wealth, commands)
    kill_by_spawn(1, 2, units, wealth, commands)
    spawn_level_1_on_border(wealth, commands)

    # BUILD
    for available_mine in set(g.mine_spots) & set(g.available_squares):
        if wealth.gold >= 20:
            commands.append(f"BUILD MINE {available_mine.x} {available_mine.y}")
            wealth.gold -= 20

    # MOVE
    enemy_hq = [b for b in buildings if b.owner == Side.THEM and b.type == BuildingType.HQ][0]
    my_hq = [b for b in buildings if b.owner == Side.ME and b.type == BuildingType.HQ][0]

    for unit in g.my_units:
        move = bfs(Point(unit.x, unit.y), Point(enemy_hq.x, enemy_hq.y))
        if move:
            commands.append(f"MOVE {unit.id} {move.x} {move.y}")

    return commands

def main():
    g.mine_spots = initial_input()
    while True:
        global start_time
        start_time = time()
        wealth, gamemap, buildings, units = turn_input()
        commands = make_move(wealth, gamemap, buildings, units)
        if commands:
            print(";".join(commands))
        else:
            print("WAIT")
        #log(start_time - time())

main()