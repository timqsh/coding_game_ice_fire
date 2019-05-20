
import sys
import math
from collections import namedtuple, deque
from enum import Enum
from random import randint, choice, shuffle

class Side(Enum):
    ME = 0
    THEM = 1

class BuildingType(Enum):
    HQ = 0
    MINE = 1

Point = namedtuple("Point", ["x", "y"])
Unit = namedtuple("Unit", ["owner", "id", "level", "x", "y"])
Building = namedtuple("Building", ["owner", "type", "x", "y"])
Wealth = namedtuple("Wealth", ["gold", "income", "opponent_gold", "opponent_income"])

def log(x):
    print(x, file=sys.stderr)

def initial_input():
    number_mine_spots = int(input())
    mine_spots = []
    for i in range(number_mine_spots):
        x, y = [int(j) for j in input().split()]
        mine_spots.append((x, y))
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

def move_random(unit):
    target = choice(neighbors(unit))
    return f"MOVE {unit.id} {target.x} {target.y}"

def neighbors(point, gamemap=None, rand=False):
    res = []
    if point.x < 11:
        res.append(Point(point.x+1, point.y))
    if point.x > 0:
        res.append(Point(point.x-1, point.y))
    if point.y < 11:
        res.append(Point(point.x, point.y+1))
    if point.y > 0:
        res.append(Point(point.x, point.y-1))

    if gamemap:
        res = [r for r in res if gamemap[r.x][r.y] != "#"]

    if rand:
        shuffle(res)

    return res

def bfs(gamemap, start, end):

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
        for pos in neighbors(cur, gamemap, rand=True):
            if pos not in prev:
                prev[pos] = cur
                q.append(pos)

def make_move(wealth, gamemap, buildings, units):
    commands=[]
    my_units=[u for u in units if u.owner == Side.ME]

    # 2.1 MOVE
    enemy_hq = [b for b in buildings if b.owner == Side.THEM and b.type == BuildingType.HQ][0]
    my_hq = [b for b in buildings if b.owner == Side.ME and b.type == BuildingType.HQ][0]

    for unit in my_units:
        #commands.append(move_random(unit))
        move = bfs(gamemap, Point(unit.x, unit.y), Point(enemy_hq.x, enemy_hq.y))
        commands.append(f"MOVE {unit.id} {move.x} {move.y}")

    # 2.2 TRAIN
    if len(my_units) <= 20:
        if my_units:
            commands.append(f"TRAIN 1 1 0")
            commands.append(f"TRAIN 1 10 11")
        else:
            commands.append(f"TRAIN 1 0 1")
            commands.append(f"TRAIN 1 11 10")
                
    return commands

def main():
    mine_spots = initial_input()
    while True:
        wealth, gamemap, buildings, units = turn_input()
        commands = make_move(wealth, gamemap, buildings, units)
        print(";".join(commands))

main()
