import numpy as np

def checkRowColumnSquare(x ,y ,notes):

    note = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    c = 0
    square = [x // 3, y // 3]

    #check row
    if len(note) > 1:
        for i in range(9):
            if not isinstance(notes[y][i], list):
                if notes[y][i] in note:
                    note.remove(notes[y][i])

    #check column
    if len(note) > 1:
        for i in range(9):
            if not isinstance(notes[i][x], list):
                if notes[i][x] in note:
                    note.remove(notes[i][x])

    #check square
    if len(note) > 1:
        for y in range(square[1]*3, square[1]*3+3):
            for x in range(square[0]*3, square[0]*3+3):
                if not isinstance(notes[y][x], list):
                    if notes[y][x] in note:
                        note.remove(notes[y][x])

    return note


unsolved = [[0,0,2,0,0,3,0,0,4],
            [3,8,0,6,0,0,0,0,7],
            [0,0,9,8,0,0,0,1,0],
            [5,0,0,4,0,0,0,0,0],
            [8,0,4,0,0,0,0,9,0],
            [0,0,0,0,0,1,7,0,0],
            [0,4,0,0,2,0,3,7,0],
            [0,9,0,0,0,0,0,0,1],
            [6,7,0,0,5,8,4,2,0]]

solved = [[7,6,2,5,1,3,9,8,4],
            [3,8,1,6,9,4,2,5,7],
            [4,5,9,8,7,2,6,1,3],
            [5,1,7,4,6,9,8,3,2],
            [8,2,4,7,3,5,1,9,6],
            [9,3,6,2,8,1,7,4,5],
            [1,4,5,9,2,6,3,7,8],
            [2,9,8,3,4,7,5,6,1],
            [6,7,3,1,5,8,4,2,9]]

count = 1
notes = []

#create notes for each empty spot
for y in range(9):
    notes.append([])
    note = [1,2,3,4,5,6,7,8,9]

    for x in range(9):

        if unsolved[y][x] == 0:
            notes[y].append(note)
        else:
            notes[y].append(unsolved[y][x])


#while 0 not in unsolved:
for x in range(1000):
    for y in range(9):
        for x in range(9):

            if unsolved[y][x] == 0:
                note = checkRowColumnSquare(x, y, notes)
                if len(note) == 1:
                    notes[y][x] = note[0]
                else:
                    notes[y][x] = note

for x in range(9):
    print(notes[x])


