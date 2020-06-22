from copy import deepcopy

import cv2
import numpy as np
import matplotlib

from Prediction import returnNumber


def showImage(img_to_show, Name):
    cv2.imshow("Image", img_to_show)
    matplotlib.image.imsave("SudokuPuzzles\\"+str(Name)+'.png', img_to_show)
    cv2.waitKey(0)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, bl, br, tr) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def sortTopLeftBottomRight(contour):
    x = contour[0] - contour[0] % 10
    y = contour[1] - contour[1] % 10

    return y, x


def findSudokuPuzzlePoints(image, imageFormat):

    # ------- PREPROCESS -------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    threshold = cv2.bitwise_not(threshold)

    #----------------------------------------

    #find contours from original image preprocessed to find sudoku square
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest countour (c) by the area
    largestContour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largestContour)

    return np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])


def contourImage(boxImage):

    empty = True
    gray = cv2.cvtColor(boxImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=3)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = 0
    if len(contours) != 1:
        x, y, w, h = cv2.boundingRect(contours[1])
        if 200 < w * h < 1200:

            ychange = int((10 - h % 10) / 2) + 2
            xchange = int(((h + 2 * ychange) - w) / 2)

            box = np.array([[x - xchange, y - ychange], [x - xchange, y + h + ychange], [x + w + xchange, y + h + ychange], [x + w + xchange, y - ychange]])
            cnt = contours[1]
            number = four_point_transform(boxImage, box)
            empty = False

    boxImage = cv2.dilate(gray, kernel, iterations=3)

    if empty:
        return boxImage, empty, cnt
    else:
        return number, empty, cnt


def createSudokuPuzzleArray(warped, imageFormat):

    # ------- PREPROCESS -------------
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    threshold = cv2.bitwise_not(threshold)

    # ----------------------------------------
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted([cv2.boundingRect(c) for c in contours], key=sortTopLeftBottomRight)

    index = 0
    puzzleArray = []
    row = []
    images_to_predict = []
    emptyLocations = []

    for cnt in cnts:
        if 4000 > cnt[2] * cnt[3] > 1000:
            index += 1
            x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
            box = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
            boxImage = four_point_transform(warped, box)

            numberImage, empty, cnt = contourImage(boxImage)

            if empty:
                row.append(0)
                emptyLocations.append((int(x+w/3), int(y+h/1.5)))
            else:
                numberImage = cv2.cvtColor(numberImage, cv2.COLOR_BGR2GRAY)
                images_to_predict.append([numberImage, False])
                row.append('x')
            if index % 9 == 0:
                puzzleArray.append(row)
                row = []

    predictions = returnNumber(images_to_predict)

    i = 0
    for x in range(len(puzzleArray)):
        for y in range(len(puzzleArray[x])):
            if puzzleArray[x][y] == 'x':
                puzzleArray[x][y] = np.argmax(predictions[i])
                i += 1

    return puzzleArray, emptyLocations


def solve(board):

    find = findEmpty(board)
    if not find:
        return True
    else:
        row, column = find

    for i in range(1,10):
        if valid(board, i, (row,column)):
            board[row][column] = i

            if solve(board):
                return True

            board[row][column] = 0

    return False


def valid(board, num, pos):

    # check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    #Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    #Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True


def findEmpty(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                return (i,j)

    return None


#path to the sudoku puzzle image
imagePath = "SudokuPuzzles/Puzzle1.png"

# read file to image variable
image = cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)

# split to find image format - png, jpeg, jpg
Prefix, imageFormat = imagePath.split(".")
Name = Prefix.split("/")[-1]
Name = Name + "Solution"

# find 4 corner points of sudoku puzzle
pts = findSudokuPuzzlePoints(image, imageFormat)

# perspective warp to get just puzzle image
warped = four_point_transform(image, pts)

# convert to 512x512 pixels
warped = cv2.resize(warped, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

# create puzzle
puzzleArray, emptyLocations = createSudokuPuzzleArray(warped, imageFormat)

# save unsolved puzzle to iterate through and refill
unsolvedPuzzleArray = deepcopy(puzzleArray)

# function to solve the puzzle
solve(puzzleArray)

index = 0
for x in range(len(unsolvedPuzzleArray)):
    for y in range(len(unsolvedPuzzleArray[x])):

        if unsolvedPuzzleArray[x][y] == 0:
            cv2.putText(warped, str(puzzleArray[x][y]), emptyLocations[index], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
            index += 1

showImage(warped, Name)

#cv2.drawContours(image, contours, -1, (0,255,0), 1)
#cv2.drawContours(image, [cnt], 0, (0,255,0), 1)
#cv2.imshow("123", boxImage)
#cv2.waitKey(0)
#x,y,w,h = cv.boundingRect(cnt)
#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


