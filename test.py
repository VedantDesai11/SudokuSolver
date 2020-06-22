x = 7
y = 2
square = [x//3, y//3]

for y in range(square[1] * 3, square[1] * 3 + 3):
    for x in range(square[0] * 3, square[0] * 3 + 3):
        print(x, y)