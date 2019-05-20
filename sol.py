
import sys
import math
from collections import namedtuple
from random import randint

Unit = namedtuple("unit", ["owner", "unit_id", "level", "x", "y"])

number_mine_spots = int(input())
for i in range(number_mine_spots):
    x, y = [int(j) for j in input().split()]

# game loop
while True:
    gold = int(input())
    income = int(input())
    opponent_gold = int(input())
    opponent_income = int(input())
    for i in range(12):
        line = input()
    building_count = int(input())
    for i in range(building_count):
        owner, building_type, x, y = [int(j) for j in input().split()]
    unit_count = int(input())
    units = []
    for i in range(unit_count):
        owner, unit_id, level, x, y = [int(j) for j in input().split()]
        units.append(Unit(owner, unit_id, level, x, y))
         
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    my_units = [u for u in units if u[0]==0]
    commands = []
    # MOVE
    for unit in my_units:
        movex = unit.x+randint(-1,1)
        movey = unit.y+randint(-1,1)
        
        if movex==-1:
            movex=0
        if movey==-1:
            movey=0
        if movex==12:
            movex=11
        if movey==12:
            movey=11
        
        commands.append(f"MOVE {unit.unit_id} {movex} {movey}")
    
    # TRAIN
    if len(my_units) <=10:
        if my_units:
            commands.append(f"TRAIN 1 1 0")
            commands.append(f"TRAIN 1 10 11")
        else:
            commands.append(f"TRAIN 1 0 1")
            commands.append(f"TRAIN 1 11 10")
            
    print(";".join(commands))
    #print("WAIT")
