
import sys
import math
from collections import namedtuple, deque, defaultdict
from enum import Enum
from random import randint, choice, shuffle, seed
from dataclasses import dataclass
from time import time
import heapq
from copy import deepcopy

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
    enemy_tower_zones: list = None
    point_min_level: dict = None # Dict[Point, int]
    my_hq: Point = None
    enemy_hq: Point = None
    enemy_units: list = None
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

def turn_input(test=False):

    if test:
        file = open("in.txt", mode="r")
        def input():
            return file.readline()

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

def neighbors(point, randomize=False, game=None):
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
    
    game = game or g
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
        try:
            weight, item = heapq.heappop(self.items)
        except Exception as e:
            log(f"Exception: {self.items}")
            raise
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

def occupied(point, gamemap, *collections):
    for collection in collections:
        for item in collection:
            if point.x == item.x and point.y == item.y:
                return True
    if gamemap[point.y][point.x] == "#":
        return True
    return False

def calculate_globals(gamemap, buildings, units):
    g.enemy_hq = [Point(b.x, b.y) for b in buildings if b.owner == Side.THEM and b.type == BuildingType.HQ][0]
    g.my_hq = [Point(b.x, b.y) for b in buildings if b.owner == Side.ME and b.type == BuildingType.HQ][0]

    g.enemy_units = [u for u in units if u.owner == Side.THEM]
    g.my_units_pos={Point(u.x, u.y) for u in units if u.owner == Side.ME}
    g.map = gamemap
  
    my_buildings = {Point(u.x, u.y) for u in buildings if u.owner == Side.ME}

    g.available_squares = set()
    my_squares = set()
    g.my_units=[u for u in units if u.owner == Side.ME]
    for x in range(12):
        for y in range(12):
            if g.map[y][x] == "O":
                for n in neighbors(Point(x, y)):
                    if not occupied(n, g.map, g.my_units, my_buildings):
                        g.available_squares.add(n)
                if not occupied(Point(x, y), g.map, g.my_units, my_buildings):
                    g.available_squares.add(Point(x, y))
                    my_squares.add(Point(x, y))
    g.border_squares = list(g.available_squares - my_squares)

    enemy_tower_pos = [Point(b.x, b.y) for b in buildings if b.owner == Side.THEM and b.type == BuildingType.TOWER]
    active_enemy_tower = [p for p in enemy_tower_pos if g.map[p.y][p.x] == "X"]
    enemy_tower_neighbors = []
    for p in active_enemy_tower:
        enemy_tower_neighbors += neighbors(p)
    enemy_tower_active_neighbors = [p for p in enemy_tower_neighbors if g.map[p.y][p.x] == "X"]
    g.enemy_tower_zones = active_enemy_tower + enemy_tower_active_neighbors
    
    point_min_level = defaultdict(lambda: 1)
    for p in g.enemy_tower_zones:
        point_min_level[p] = 3
    enemy_units = [u for u in units if u.owner == Side.THEM]
    for u in enemy_units:
         level_to_beat_unit = u.level + 1 if u.level < 3 else 3
         p = Point(u.x, u.y)
         point_min_level[p] = max(point_min_level[p], level_to_beat_unit)
    g.point_min_level = point_min_level

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
    while wealth.gold >= 10 and wealth.income >= 0 and g.border_squares and len(g.my_units_pos) < 10:
        spawn_point = choice(g.border_squares)
        commands.append(f"TRAIN 1 {spawn_point.x} {spawn_point.y}")
        wealth.gold -= 10
        wealth.income -= 1
        g.border_squares.remove(spawn_point)

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

def dfs_enemy(start, g_new:G, debug=False):
    # if debug:
    #     log(f"dfs: {start}")
    g_new.map[start.y][start.x] = "X"
    for n in enemy_neighbors(start, g_new):
        dfs_enemy(n, g_new, debug)

def calc_turn(p:Point, g_old):
    # debug = p == Point(3,8)
    g_new = G()
    #g.border_squares TODO

    g_new.my_units_pos = deepcopy(g_old.my_units_pos)
    g_new.my_units_pos.add(p)
    g_new.enemy_hq = g_old.enemy_hq

    g_new.map = []
    for row in g_old.map:
        g_new.map.append( ["x" if c=="X" else c for c in row] )
    # if debug:
    #     log(f"old   : {g_new.map}")
    g_new.map[p.y][p.x] = "O"
    # if debug:
    #     log(f"before: {g_new.map}")
    dfs_enemy(g_old.enemy_hq, g_new, debug=False)
    # if debug:
    #     log(f"after : {g_new.map}")

    g_new.enemy_units = []
    for u in g_old.enemy_units:
        if g_new.map[u.y][u.x] == "X":
            g_new.enemy_units.append(Unit(*u))

    return g_new

def units_cost(units):
    return sum(recruitment_cost(u.level) for u in units)

def map_count(char, game):
    return sum(row.count(char) for row in game.map)

def try_cut():
    CELL_FACTOR = 2
    profit = {}
    for b in g.border_squares:
        # TODO neighbors without my squares
        cost1 = recruitment_cost(g.point_min_level[b])
        g_new = calc_turn(b, g)
        unit_gain1 = units_cost(g.enemy_units) - units_cost(g_new.enemy_units)
        map_gain1 = CELL_FACTOR * (map_count("X", g) - map_count("X", g_new))
        profit[(b,)] = map_gain1 + unit_gain1 - cost1
        # log(f"{b}: {map_gain} + {unit_gain} - {cost}")
        for n1 in neighbors(b):
            cost2 = cost1 + recruitment_cost(g.point_min_level[n1])
            g_new_2 = calc_turn(n1, g_new)
            unit_gain2 = unit_gain1 + units_cost(g_new.enemy_units) - units_cost(g_new_2.enemy_units)
            map_gain2 = map_gain1 + CELL_FACTOR * (map_count("X", g_new) - map_count("X", g_new_2))
            profit[(b, n1)] = map_gain2 + unit_gain2 - cost2
            for n2 in neighbors(n1):
                cost3 = cost2 + recruitment_cost(g.point_min_level[n2])
                g_new_3 = calc_turn(n2, g_new_2)
                unit_gain3 = unit_gain2 + units_cost(g_new_2.enemy_units) - units_cost(g_new_3.enemy_units)
                map_gain3 = map_gain2 + CELL_FACTOR * (map_count("X", g_new_2) - map_count("X", g_new_3))
                profit[(b, n1, n2)] = map_gain3 + unit_gain3 - cost3
                for n3 in neighbors(n2):
                    cost4 = cost3 + recruitment_cost(g.point_min_level[n3])
                    g_new_4 = calc_turn(n3, g_new_3)
                    unit_gain4 = unit_gain3 + units_cost(g_new_3.enemy_units) - units_cost(g_new_4.enemy_units)
                    map_gain4 = map_gain3 + CELL_FACTOR * (map_count("X", g_new_3) - map_count("X", g_new_4))
                    profit[(b, n1, n2, n3)] = map_gain4 + unit_gain4 - cost4

    best_moves = max(profit, key=profit.get, default=0)
    if profit[best_moves] > 0:
        return best_moves
    else:
        return []

def make_move(wealth, gamemap, buildings, units):
    commands=[]

    calculate_globals(gamemap, buildings, units)
    
    # TRY INSTANT KILL
    solution, cost = dijkstra(g.enemy_hq, g.border_squares)
    if cost <= wealth.gold:
        for elem in solution:
            commands.append(f"TRAIN {elem.level} {elem.x} {elem.y}")
        return commands

    # TRY CUT
    best_moves = try_cut()
    if best_moves:
        for point in best_moves:
            commands.append(f"TRAIN {g.point_min_level[point]} {point.x} {point.y}")

    # TRAIN
    #kill_by_spawn(3, 3, units, wealth, commands)
    #kill_by_spawn(2, 3, units, wealth, commands)
    #kill_by_spawn(1, 2, units, wealth, commands)
    spawn_level_1_on_border(wealth, commands)

    # BUILD
    # for available_mine in set(g.mine_spots) & set(g.available_squares):
    #     if wealth.gold >= 20:
    #         commands.append(f"BUILD MINE {available_mine.x} {available_mine.y}")
    #         wealth.gold -= 20

    # MOVE
    for unit in g.my_units:
        move = bfs(Point(unit.x, unit.y), g.enemy_hq)
        if move:
            commands.append(f"MOVE {unit.id} {move.x} {move.y}")

    return commands

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        wealth, gamemap, buildings, units = turn_input(test=True)
        commands = make_move(wealth, gamemap, buildings, units)
        print(";".join(commands))
        return
    ###
    g.mine_spots = initial_input()
    while True:
        global start_time
        start_time = time()
        wealth, gamemap, buildings, units = turn_input()
        log(wealth)
        log(gamemap)
        log(buildings)
        log(units)
        commands = make_move(wealth, gamemap, buildings, units)
        if commands:
            print(";".join(commands))
        else:
            print("WAIT")
        #log(start_time - time())

main()