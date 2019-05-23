
import sys
import math
from collections import namedtuple, deque, defaultdict
from enum import Enum
from random import randint, choice, shuffle, seed
from dataclasses import dataclass
from time import time
import heapq
from copy import copy, deepcopy

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
    mine_spots: list = None
        
    my_hq: Point = None
    enemy_hq: Point = None

    map: list = None
    units: list = None
    buildings: list = None
    wealth: Wealth = None
    
    enemy_units: list = None
    my_units: list = None
    my_units_pos: set = None

    border_squares: list = None
    available_squares: list = None
    
    enemy_tower_zones: list = None
    point_min_level: dict = None # Dict[Point, int]

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

def turn_input(input):

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

def neighbors(point, game=None, notmine=False):
    res = []
    game = game or g
    if g.enemy_hq.x == 0:
        directions = [(-1,0),(0,-1),(1,0),(0,1)]
    else:
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
    for d in directions:
        new_x = point.x + d[0]
        new_y = point.y + d[1]
        if 0 <= new_x <= 11 and 0 <= new_y <= 11:
            res.append(Point(new_x, new_y))
    if notmine:
        res = [r for r in res if g.map[r.y][r.x] != "O"]    
    res = [r for r in res if g.map[r.y][r.x] != "#"]
    res = [r for r in res if Point(r.x, r.y) not in game.my_units_pos]
    return res

class PriorityQueue:
    def __init__(self, items=None):
        self.items = items or []
        heapq.heapify(self.items)

    def __bool__(self):
        return bool(self.items)

    def push(self, item, weight):
        heapq.heappush(self.items, (weight, item))

    def pop(self):
        weight, item = heapq.heappop(self.items)
        return item, weight

def dijkstra(target, starting_points):
    
    SolutionNode = namedtuple("SolutionNode", ["x", "y", "level"])
    AlgorithmNode = namedtuple("AlgorithmNode", ["cost", "prev"])
    
    def form_solution(prev, cur):
        solution = [SolutionNode(cur.x, cur.y, g.point_min_level[cur])]
        while prev[cur].prev:
            cur_node = prev[cur]
            cur_point = cur_node.prev
            solution.append(SolutionNode(cur_point.x, cur_point.y, g.point_min_level[cur_point]))
            cur = cur_point
        cost = sum(recruitment_cost(n.level) for n in solution)
        return solution, cost 

    q = PriorityQueue()
    q.push(target, 0)
    prev = {target: AlgorithmNode(cost=0, prev=None)}
    while q:
        cur, cur_weight = q.pop()
        if cur in starting_points:
            return form_solution(prev, cur)
        for pos in neighbors(cur):
            prev_cost, prev_point = prev[cur]
            if pos not in prev:
                add_cost = recruitment_cost(g.point_min_level[pos])
                prev[pos] = AlgorithmNode(cost=prev_cost+add_cost, prev=cur)
                q.push(pos, prev_cost+add_cost)
    return None, math.inf

def choose_move(unit):
    prior = {
        "X":0,
        "x":1,
        "o":2,
        ".":3,
        "O":4,
    }
    p = Point(unit.x, unit.y)
    available_neighbors = [n for n in neighbors(p) if g.point_min_level[n] <= unit.level]
    return min(available_neighbors, key=lambda p: prior[g.map[p.y][p.x]], default=None)

def occupied(point, gamemap, *collections):
    for collection in collections:
        for item in collection:
            if point.x == item.x and point.y == item.y:
                return True
    if gamemap[point.y][point.x] == "#":
        return True
    return False

def calc_border_squares(game):
    my_buildings = {Point(u.x, u.y) for u in game.buildings if u.owner == Side.ME}
    game.available_squares = set()
    my_squares = set()
    for x in range(12):
        for y in range(12):
            if game.map[y][x] == "O":
                for n in neighbors(Point(x, y)):
                    if not occupied(n, game.map, game.my_units, my_buildings):
                        game.available_squares.add(n)
                if not occupied(Point(x, y), game.map, game.my_units, my_buildings):
                    game.available_squares.add(Point(x, y))
                    my_squares.add(Point(x, y))
    game.border_squares = list(game.available_squares - my_squares)

def calculate_globals():
    g.enemy_hq = [Point(b.x, b.y) for b in g.buildings if b.owner == Side.THEM and b.type == BuildingType.HQ][0]
    g.my_hq = [Point(b.x, b.y) for b in g.buildings if b.owner == Side.ME and b.type == BuildingType.HQ][0]

    g.my_units=[u for u in g.units if u.owner == Side.ME]
    g.enemy_units = [u for u in g.units if u.owner == Side.THEM]
    g.my_units_pos={Point(u.x, u.y) for u in g.units if u.owner == Side.ME}

    calc_border_squares(g)

    enemy_tower_pos = [Point(b.x, b.y) for b in g.buildings if b.owner == Side.THEM and b.type == BuildingType.TOWER]
    active_enemy_tower = [p for p in enemy_tower_pos if g.map[p.y][p.x] == "X"]
    enemy_tower_neighbors = []
    for p in active_enemy_tower:
        enemy_tower_neighbors += neighbors(p)
    enemy_tower_active_neighbors = [p for p in enemy_tower_neighbors if g.map[p.y][p.x] == "X"]
    g.enemy_tower_zones = active_enemy_tower + enemy_tower_active_neighbors
    
    point_min_level = defaultdict(lambda: 1)
    for p in g.enemy_tower_zones:
        point_min_level[p] = 3
    for u in g.enemy_units:
         level_to_beat_unit = u.level + 1 if u.level < 3 else 3
         p = Point(u.x, u.y)
         point_min_level[p] = max(point_min_level[p], level_to_beat_unit)
    g.point_min_level = point_min_level

def enemy_neighbors(point, g_new):
    res = []
    if point.x < 11:
        res.append(Point(point.x+1, point.y))
    if point.x > 0:
        res.append(Point(point.x-1, point.y))
    if point.y < 11:
        res.append(Point(point.x, point.y+1))
    if point.y > 0:
        res.append(Point(point.x, point.y-1))

    res = [r for r in res if g_new.map[r.y][r.x] == "x"]

    return res

def dfs_enemy(start, g_new:G):
    g_new.map[start.y][start.x] = "X"
    for n in enemy_neighbors(start, g_new):
        dfs_enemy(n, g_new)

def calc_spawn(p: Point, g_old):
    g_new = G()
    g_new.my_hq = g_old.my_hq
    g_new.enemy_hq = g_old.enemy_hq

    g_new.map = []
    for row in g_old.map:
        g_new.map.append( ["x" if c=="X" else c for c in row] )
    g_new.map[p.y][p.x] = "O"
    dfs_enemy(g_old.enemy_hq, g_new)
    # TODO dfs_my (inactive to active)
    
    g_new.enemy_units = []
    for u in g_old.enemy_units:
        if g_new.map[u.y][u.x] == "X":
            g_new.enemy_units.append(Unit(*u))

    return g_new

def calc_move(unit:Unit, dest:Point, g_old:G):
    g_new = G()
    g_new.mine_spots = g_old.mine_spots
    g_new.my_hq = g_old.my_hq
    g_new.enemy_hq = g_old.enemy_hq
    
    g_new.units = copy(g_old.units)
    g_new.units.remove(unit)
    g_new.units.append(Unit(owner=Side.ME, id=unit.id, level=unit.level, x=dest.x, y=dest.y))

    g_new.my_units = copy(g_old.my_units)
    g_new.my_units.remove(unit)
    g_new.my_units.append(Unit(owner=Side.ME, id=unit.id, level=unit.level, x=dest.x, y=dest.y))

    g_new.buildings = copy(g_old.buildings)
    g_new.wealth = copy(g_old.wealth)

    g_new.enemy_tower_zones = copy(g_old.enemy_tower_zones)
    g_new.point_min_level = copy(g_old.point_min_level)
    
    g_new.map = []
    for row in g_old.map:
        g_new.map.append( ["x" if c=="X" else c for c in row] )
    g_new.map[dest.y][dest.x] = "O"
    dfs_enemy(g_old.enemy_hq, g_new)
    # TODO dfs_my (inactive to active) 
    
    calc_border_squares(g_new)

    g_new.my_units_pos = copy(g_old.my_units_pos)
    g_new.my_units_pos.remove(Point(unit.x, unit.y))
    g_new.my_units_pos.add(dest)

    # Они уйдут только на следующем ходу
    g_new.enemy_units = copy(g_old.enemy_units)
    # for u in g_old.enemy_units:
    #     if g_new.map[u.y][u.x] == "X":
    #         g_new.enemy_units.append(Unit(*u))

    return g_new

def kill_by_spawn(enemy_level, my_level, wealth, commands):
    enemy_positions = {Point(u.x, u.y) for u in g.enemy_units}
    available_enemies = [s for s in g.border_squares if s in enemy_positions]
    while wealth.gold >= recruitment_cost(my_level) and wealth.income >= 0 and available_enemies:
        spawn_point = choice(available_enemies)
        commands.append(f"TRAIN {my_level} {spawn_point.x} {spawn_point.y}")
        wealth.gold -= recruitment_cost(my_level)
        wealth.income -= upkeep_cost(my_level)
        available_enemies.remove(spawn_point)

def spawn_level_1_on_border(wealth, commands):    
    while wealth.gold >= 10 and wealth.income >= 0 and g.border_squares and len(g.my_units_pos) < 10:
        spawn_point = choice(g.border_squares)
        commands.append(f"TRAIN 1 {spawn_point.x} {spawn_point.y}")
        wealth.gold -= 10
        wealth.income -= 1
        g.border_squares.remove(spawn_point)

def units_cost(units):
    return sum(recruitment_cost(u.level) for u in units)

def map_count(char, game):
    return sum(row.count(char) for row in game.map)

def try_cut(budget):
    CELL_FACTOR = 2
    UPKEEP_FACTOR = 2
    price = {():0}
    profit = {():0}
    worlds = {():g}
    q = deque((b,) for b in g.border_squares)
    while q:
        moves = q.popleft()
        need_lvl = g.point_min_level[moves[0]]
        price[moves] = price[moves[1:]] + recruitment_cost(need_lvl)
        if price[moves] > budget:
            continue
        old_world = worlds[moves[1:]]
        new_world = calc_spawn(moves[0], old_world)
        worlds[moves] = new_world
        unit_gain = units_cost(old_world.enemy_units) - units_cost(new_world.enemy_units)
        map_gain = CELL_FACTOR * (map_count("X", old_world) - map_count("X", new_world))
        cost = recruitment_cost(need_lvl) + UPKEEP_FACTOR * upkeep_cost(need_lvl)
        profit[moves] = profit[moves[1:]] + map_gain + unit_gain - cost
        if len(moves) < 3:
            for n in neighbors(moves[0], notmine=True):
                q.append(tuple([n,*moves]))
    if profit:
        best_moves = max(profit, key=profit.get)
        if profit[best_moves] > 0:
            log(f"profit={profit[best_moves]}, {len(best_moves)} turns")
            return reversed(best_moves)
    return []

def make_move():
    global g
    commands=[]
    calculate_globals()

    # MOVE
    for unit in g.my_units:
        move = choose_move(unit)
        if move:
            g = calc_move(unit, move, g)
            commands.append(f"MOVE {unit.id} {move.x} {move.y}")

    # TRY INSTANT KILL
    solution, cost = dijkstra(g.enemy_hq, g.border_squares)
    if cost <= g.wealth.gold and solution:
        for elem in solution:
            commands.append(f"TRAIN {elem.level} {elem.x} {elem.y}")
        return commands

    # TRY CUT
    best_moves = try_cut(g.wealth.gold)
    if best_moves:
        for point in best_moves:
            g_new = calc_spawn(point, g)
            g.enemy_units = g_new.enemy_units
            g.wealth.gold -= recruitment_cost(g.point_min_level[point])
            commands.append(f"TRAIN {g.point_min_level[point]} {point.x} {point.y}")

    # TRAIN
    #kill_by_spawn(3, 3, wealth, commands)
    #kill_by_spawn(2, 3, wealth, commands)
    kill_by_spawn(1, 2, g.wealth, commands)
    spawn_level_1_on_border(g.wealth, commands)

    # BUILD MINES
    # for available_mine in set(g.mine_spots) & set(g.available_squares):
    #     if wealth.gold >= 20:
    #         commands.append(f"BUILD MINE {available_mine.x} {available_mine.y}")
    #         wealth.gold -= 20

    return commands

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        file = open("in.txt", mode="r")
        def test_input():
            return file.readline()
        g.wealth, g.map, g.buildings, g.units = turn_input(test_input)
        commands = make_move()
        print(";".join(commands))
        return
    ###
    g.mine_spots = initial_input()
    while True:
        global start_time
        start_time = time()
        g.wealth, g.map, g.buildings, g.units = turn_input(input)
        commands = make_move()
        if commands:
            print(";".join(commands))
        else:
            print("WAIT")
        #log(start_time - time())

main()