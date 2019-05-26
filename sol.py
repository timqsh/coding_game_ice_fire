import heapq
import math
import random
import sys
from collections import defaultdict, deque, namedtuple
from typing import NamedTuple, Dict, List, Set, Any
from copy import copy
from dataclasses import dataclass
from time import time

random.seed(1337)
start_time = time()

ME = 0
THEM = 1
HQ = 0
MINE = 1
TOWER = 2


class Pos(NamedTuple):
    x: int
    y: int


class Unit(NamedTuple):
    owner: int
    id: int
    level: int
    x: int
    y: int


class Building(NamedTuple):
    owner: int
    type: int
    x: int
    y: int


@dataclass
class Wealth:
    gold: int
    income: int
    opponent_gold: int
    opponent_income: int


@dataclass
class Game:
    mine_spots: List[Unit] = None

    my_hq: Pos = None
    enemy_hq: Pos = None

    map: List[List[str]] = None
    units: List[Unit] = None
    buildings: List[Building] = None
    wealth: Wealth = None

    enemy_units: List[Unit] = None
    my_units: List[Unit] = None
    my_units_pos: Set[Pos] = None

    border_positions: List[Pos] = None
    available_positions: List[Pos] = None

    enemy_tower_zones: List[Pos] = None
    pos_min_level: Dict[Pos, int] = None


g = Game()
chat_message = ""


def log(x: Any):
    print(x, file=sys.stderr)


def msg(s: str):
    global chat_message
    chat_message = chat_message + ("|" if chat_message else "") + s


def recruitment_cost(level):
    return [10, 20, 30][level - 1]


def upkeep_cost(level):
    return [1, 4, 20][level - 1]


def initial_input(input):
    number_mine_spots = int(input())
    mine_spots = []
    for i in range(number_mine_spots):
        x, y = [int(j) for j in input().split()]
        mine_spots.append(Pos(x, y))
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
        buildings.append(Building(*params))

    unit_count = int(input())
    units = []
    for i in range(unit_count):
        params = [int(j) for j in input().split()]
        units.append(Unit(*params))

    return wealth, gamemap, buildings, units


def position_playable(p, game):
    return 0 <= p.x <= 11 and 0 <= p.y <= 11 and game.map[p.y][p.x] != "#"


def my_positions(game):
    result = []
    for x in range(12):
        for y in range(12):
            p = Pos(x, y)
            if my_pos(p, game):
                result.append(p)
    return result


def my_pos(p, game):
    return game.map[p.y][p.x] == "O"


def my_unit(p, game):
    return p in game.my_units_pos


def neighbors(pos, game=None, notmine=False):
    res = []
    game = game or g
    if game.enemy_hq.x == 0:
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    else:
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for d in directions:
        p = Pos(pos.x + d[0], pos.y + d[1])
        if not position_playable(p, game):
            continue
        if (notmine and my_pos(p, game)) or my_unit(p, game):
            continue
        res.append(p)
    return res


def allneighbors(pos, game):
    res = []
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for d in directions:
        p = Pos(pos.x + d[0], pos.y + d[1])
        if not position_playable(p, game):
            continue
        res.append(p)
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


def dijkstra(target, starting_positions, game: Game):
    SolutionNode = namedtuple("SolutionNode", ["x", "y", "level"])
    AlgorithmNode = namedtuple("AlgorithmNode", ["cost", "prev"])

    def form_solution(prev, cur):
        solution = [SolutionNode(cur.x, cur.y, game.pos_min_level[cur])]
        while prev[cur].prev:
            cur_node = prev[cur]
            cur_pos = cur_node.prev
            solution.append(
                SolutionNode(cur_pos.x, cur_pos.y, game.pos_min_level[cur_pos])
            )
            cur = cur_pos
        cost = sum(recruitment_cost(n.level) for n in solution)
        return solution, cost

    q = PriorityQueue()
    q.push(target, 0)
    prev = {target: AlgorithmNode(cost=0, prev=None)}
    while q:
        cur, cur_weight = q.pop()
        if cur in starting_positions:
            return form_solution(prev, cur)
        for pos in neighbors(cur, game):
            prev_cost, prev_pos = prev[cur]
            if pos not in prev:
                add_cost = recruitment_cost(game.pos_min_level[pos])
                prev[pos] = AlgorithmNode(cost=prev_cost + add_cost, prev=cur)
                q.push(pos, prev_cost + add_cost)
    return None, math.inf


def dist_chebyshev(p):
    return max(abs(g.enemy_hq.x - p.x), abs(g.enemy_hq.y - p.y))


def choose_move(unit):
    prior = {"X": 0, "x": 1, "o": 2, ".": 3, "O": 4}
    p = Pos(unit.x, unit.y)
    all_neighbors = neighbors(p)
    available_neighbors = [n for n in all_neighbors if g.pos_min_level[n] <= unit.level]

    # stand your ground
    enemies_near = len(all_neighbors) > len(available_neighbors)
    if enemies_near:
        available_neighbors = [n for n in available_neighbors if not my_pos(n, g)]

    available_neighbors.sort(key=dist_chebyshev)
    return min(available_neighbors, key=lambda p: prior[g.map[p.y][p.x]], default=None)


def occupied(pos, gamemap, *collections):
    for collection in collections:
        for item in collection:
            if pos.x == item.x and pos.y == item.y:
                return True
    return False


def calc_border_positions(game):
    my_buildings_pos = {Pos(u.x, u.y) for u in game.buildings if u.owner == ME}
    game.available_positions = set()
    my_available_positions = set()
    for p in my_positions(game):
        for n in neighbors(p):
            if not occupied(n, game.map, game.my_units, my_buildings_pos):
                game.available_positions.add(n)
        if not occupied(p, game.map, game.my_units, my_buildings_pos):
            game.available_positions.add(p)
            my_available_positions.add(p)
    game.border_positions = list(game.available_positions - my_available_positions)


def calculate_globals(game):
    game.enemy_hq = [
        Pos(b.x, b.y) for b in game.buildings if b.owner == THEM and b.type == HQ
    ][0]
    game.my_hq = [
        Pos(b.x, b.y) for b in game.buildings if b.owner == ME and b.type == HQ
    ][0]

    game.my_units = [u for u in game.units if u.owner == ME]
    game.enemy_units = [u for u in game.units if u.owner == THEM]
    game.my_units_pos = {Pos(u.x, u.y) for u in game.units if u.owner == ME}

    calc_border_positions(game)

    enemy_tower_pos = [
        Pos(b.x, b.y) for b in game.buildings if b.owner == THEM and b.type == TOWER
    ]
    active_enemy_tower = [p for p in enemy_tower_pos if game.map[p.y][p.x] == "X"]
    enemy_tower_neighbors = []
    for p in active_enemy_tower:
        enemy_tower_neighbors += neighbors(p)
    enemy_tower_active_neighbors = [
        p for p in enemy_tower_neighbors if game.map[p.y][p.x] == "X"
    ]
    game.enemy_tower_zones = active_enemy_tower + enemy_tower_active_neighbors

    pos_min_level = defaultdict(lambda: 1)
    for p in game.enemy_tower_zones:
        pos_min_level[p] = 3
    for u in game.enemy_units:
        level_to_beat_unit = u.level + 1 if u.level < 3 else 3
        p = Pos(u.x, u.y)
        pos_min_level[p] = max(pos_min_level[p], level_to_beat_unit)
    game.pos_min_level = pos_min_level


def enemy_neighbors(pos: Pos, game: Game):
    res = []
    if pos.x < 11:
        res.append(Pos(pos.x + 1, pos.y))
    if pos.x > 0:
        res.append(Pos(pos.x - 1, pos.y))
    if pos.y < 11:
        res.append(Pos(pos.x, pos.y + 1))
    if pos.y > 0:
        res.append(Pos(pos.x, pos.y - 1))

    res = [r for r in res if game.map[r.y][r.x] == "x"]
    return res


def dfs_enemy(start, game: Game):
    game.map[start.y][start.x] = "X"
    for n in enemy_neighbors(start, game):
        dfs_enemy(n, game)


def dfs_my(start, game: Game):
    game.map[start.y][start.x] = "O"
    for n in allneighbors(start, game):
        if game.map[n.y][n.x] == "o":
            dfs_my(n, game)


def calc_spawn(p: Pos, g_old):
    g_new = Game()
    g_new.my_hq = g_old.my_hq
    g_new.enemy_hq = g_old.enemy_hq

    g_new.map = []
    for row in g_old.map:
        g_new.map.append(["x" if c == "X" else c for c in row])
    g_new.map[p.y][p.x] = "O"
    dfs_enemy(g_old.enemy_hq, g_new)
    # TODO dfs_my (inactive to active) чтобы точнее оценивать профит

    g_new.enemy_units = []
    for u in g_old.enemy_units:
        if g_new.map[u.y][u.x] == "X":
            g_new.enemy_units.append(Unit(*u))

    g_new.my_units_pos = copy(g_old.my_units_pos)
    g_new.my_units_pos.add(p)

    g_new.units = copy(g_old.units)
    g_new.units.append(Unit(owner=ME, id=777, level=1, x=p.x, y=p.y))

    return g_new


def calc_move(unit: Unit, dest: Pos, g_old: Game):
    g_new = Game()
    g_new.mine_spots = g_old.mine_spots
    g_new.my_hq = g_old.my_hq
    g_new.enemy_hq = g_old.enemy_hq

    g_new.units = copy(g_old.units)
    g_new.units.remove(unit)
    g_new.units.append(Unit(owner=ME, id=unit.id, level=unit.level, x=dest.x, y=dest.y))

    g_new.my_units = copy(g_old.my_units)
    g_new.my_units.remove(unit)
    g_new.my_units.append(
        Unit(owner=ME, id=unit.id, level=unit.level, x=dest.x, y=dest.y)
    )

    g_new.buildings = copy(g_old.buildings)
    g_new.wealth = copy(g_old.wealth)

    g_new.enemy_tower_zones = copy(g_old.enemy_tower_zones)
    g_new.pos_min_level = copy(g_old.pos_min_level)

    g_new.map = []
    for row in g_old.map:
        g_new.map.append(["x" if c == "X" else c for c in row])
    g_new.map[dest.y][dest.x] = "O"
    dfs_enemy(g_old.enemy_hq, g_new)
    # TODO dfs_my (inactive to active) чтобы после хода было больше опций для вызова

    for x, row in enumerate(g_new.map):
        g_new.map[x] = ["o" if c == "O" else c for c in row]
    dfs_my(g_old.my_hq, g_new)  

    calc_border_positions(g_new)

    g_new.my_units_pos = copy(g_old.my_units_pos)
    g_new.my_units_pos.remove(Pos(unit.x, unit.y))
    g_new.my_units_pos.add(dest)

    # Они уйдут только на следующем ходу
    g_new.enemy_units = copy(g_old.enemy_units)
    # for u in g_old.enemy_units:
    #     if g_new.map[u.y][u.x] == "X":
    #         g_new.enemy_units.append(Unit(*u))

    return g_new


def kill_by_spawn(enemy_level, my_level, wealth, commands):
    enemy_positions = {Pos(u.x, u.y) for u in g.enemy_units}
    available_enemies = [s for s in g.border_positions if s in enemy_positions]
    while (
        wealth.gold >= recruitment_cost(my_level)
        and wealth.income >= 0
        and available_enemies
    ):
        spawn_pos = random.choice(available_enemies)
        commands.append(f"TRAIN {my_level} {spawn_pos.x} {spawn_pos.y}")
        wealth.gold -= recruitment_cost(my_level)
        wealth.income -= upkeep_cost(my_level)
        available_enemies.remove(spawn_pos)


def dist_to_enemy_hq(p: Pos):
    return abs(p.x - g.enemy_hq.x) + abs(p.y - g.enemy_hq.y)


def units_cost(units):
    return sum(recruitment_cost(u.level) for u in units)


def map_count(char, game):
    return sum(row.count(char) for row in game.map)


def try_cut_curve(budget):
    CELL_FACTOR = 2
    UPKEEP_FACTOR = 2
    price = {(): 0}
    profit = {(): 0}
    worlds = {(): g}
    q = deque((b,) for b in g.border_positions)
    while q:
        moves = q.popleft()
        need_lvl = g.pos_min_level[moves[0]]
        price[moves] = price[moves[1:]] + recruitment_cost(need_lvl)
        if price[moves] > budget:
            continue
        old_world = worlds[moves[1:]]
        new_world = calc_spawn(moves[0], old_world)
        worlds[moves] = new_world
        unit_gain = units_cost(old_world.enemy_units) - units_cost(
            new_world.enemy_units
        )
        map_gain = CELL_FACTOR * (map_count("X", old_world) - map_count("X", new_world))
        cost = recruitment_cost(need_lvl) + UPKEEP_FACTOR * upkeep_cost(need_lvl)
        profit[moves] = profit[moves[1:]] + map_gain + unit_gain - cost
        if len(moves) < 3:
            for n in neighbors(moves[0], notmine=True):
                q.append(tuple([n, *moves]))
    if profit:
        best_moves = max(profit, key=profit.get)
        if profit[best_moves] > 0:
            msg(f"{profit[best_moves]}|{len(best_moves)}")
            return reversed(best_moves)
    return []


def try_cut_straight(budget):
    CELL_FACTOR = 2
    UPKEEP_FACTOR = 2
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    price = {(): 0}
    profit = {(): 0}
    worlds = {(): g}
    q = deque(((b,), d) for b in g.border_positions for d in directions)
    while q:
        moves, direction = q.popleft()
        need_lvl = g.pos_min_level[moves[0]]
        price[moves] = price[moves[1:]] + recruitment_cost(need_lvl)
        if price[moves] > budget:
            continue
        old_world = worlds[moves[1:]]
        if moves not in worlds:
            new_world = calc_spawn(moves[0], old_world)
            worlds[moves] = new_world
        unit_gain = units_cost(old_world.enemy_units) - units_cost(
            new_world.enemy_units
        )
        map_gain = CELL_FACTOR * (map_count("X", old_world) - map_count("X", new_world))
        cost = recruitment_cost(need_lvl) + UPKEEP_FACTOR * upkeep_cost(need_lvl)
        profit[moves] = profit[moves[1:]] + map_gain + unit_gain - cost

        if len(moves) < 6:
            n = Pos(moves[0].x + direction[0], moves[0].y + direction[1])
            if position_playable(n, g) and not my_pos(n, g):
                q.append((tuple([n, *moves]), direction))

    if profit:
        best_moves = max(profit, key=profit.get)
        if profit[best_moves] > 0:
            msg(f"{profit[best_moves]}|{len(best_moves)}")
            return reversed(best_moves)
    return []


def move_units(commands):
    global g
    g.my_units.sort(key=dist_chebyshev)
    for unit in g.my_units:
        move = choose_move(unit)
        if move:
            g = calc_move(unit, move, g)
            commands.append(f"MOVE {unit.id} {move.x} {move.y}")


def try_instant_kill(commands):
    global g
    solution, cost = dijkstra(g.enemy_hq, g.border_positions, g)
    if cost <= g.wealth.gold and solution:
        msg("^_^")
        for elem in solution:
            commands.append(f"TRAIN {elem.level} {elem.x} {elem.y}")
        return True
    return False


def try_cut(commands):
    global g
    # best_moves = try_cut_curve(g.wealth.gold)
    best_moves = try_cut_straight(g.wealth.gold)
    if best_moves:
        for pos in best_moves:
            g_new = calc_spawn(pos, g)
            g.enemy_units = g_new.enemy_units
            g.wealth.gold -= recruitment_cost(g.pos_min_level[pos])
            g.units = g_new.units
            g.my_units_pos = g_new.my_units_pos
            commands.append(f"TRAIN {g.pos_min_level[pos]} {pos.x} {pos.y}")


def try_kill_by_spawn(commands):
    global g
    kill_by_spawn(3, 3, g.wealth, commands)
    kill_by_spawn(2, 3, g.wealth, commands)
    kill_by_spawn(1, 2, g.wealth, commands)


def spawn_level_1_on_border(commands):
    global g
    i = 0
    spawn_options = [p for p in g.border_positions if g.pos_min_level[p] == 1]
    while (
        g.wealth.gold >= 10
        and g.wealth.income >= 0
        and spawn_options
        and len(g.my_units_pos) + i < 5
    ):
        spawn_pos = min(spawn_options, key=dist_to_enemy_hq)
        commands.append(f"TRAIN 1 {spawn_pos.x} {spawn_pos.y}")
        g.wealth.gold -= 10
        g.wealth.income -= 1
        spawn_options.remove(spawn_pos)
        i += 1


def build_mines(commands):
    global g
    for available_mine in set(g.mine_spots) & set(g.available_positions):
        if g.wealth.gold >= 20:
            commands.append(f"BUILD MINE {available_mine.x} {available_mine.y}")
            g.wealth.gold -= 20


def build_towers(commands):
    global g
    building_pos = {Pos(b.x, b.y) for b in g.buildings}
    enemy_unit_pos = {Pos(b.x, b.y) for b in g.enemy_units}

    tower_options = [
        p
        for p in my_positions(g)
        if p not in g.my_units_pos and p not in g.mine_spots and p not in building_pos
    ]
    tower_options_weight = {t: 0 for t in tower_options}
    for p in tower_options:
        for n in allneighbors(p, g):
            if n in g.my_units_pos:
                enemy_near_my_unit = False
                for nn in allneighbors(n, g):
                    if nn in enemy_unit_pos:
                        enemy_near_my_unit = True
                tower_options_weight[p] += 2 if enemy_near_my_unit else 1
            if n in enemy_unit_pos:
                tower_options_weight[p] += 1.5

    def sort_key(p):
        return (tower_options_weight.get(p), -dist_to_enemy_hq(p))

    if g.wealth.gold >= 15 and tower_options_weight:
        p = max(tower_options_weight, key=sort_key)
        commands.append(f"BUILD TOWER {p.x} {p.y}")


def strategy():
    global g
    commands = []
    calculate_globals(g)

    move_units(commands)
    if try_instant_kill(commands):
        return commands
    try_cut(commands)
    # try_kill_by_spawn(commands)
    spawn_level_1_on_border(
        commands
    )  # TODO ставить так чтобы не мешать на след. ходу своим
    # build_mines(commands)
    build_towers(commands)

    return commands


def main():
    global start_time
    global chat_message
    is_test = len(sys.argv) == 2 and sys.argv[1] == "test"
    if is_test:
        file = open("in.txt", mode="r")

        def test_input():
            return file.readline()

    g.mine_spots = initial_input(test_input if is_test else input)
    while True:
        chat_message = ""
        start_time = time()
        g.wealth, g.map, g.buildings, g.units = turn_input(
            test_input if is_test else input
        )
        commands = strategy()
        msg(f"{time()-start_time:.2f}")
        if chat_message:
            commands.append(f"MSG {chat_message}")
        if commands:
            print(";".join(commands))
        else:
            print("WAIT")


main()
