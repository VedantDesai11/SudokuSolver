# SudokuSolver
Upload a sudoku puzzle image to the program and it outputs solved image. Works on screenshots taken from websudoku.com.

###Steps
  -Detect the puzzle from the image and crop it out. 
  -CNN used for number prediction trained on MNIST Dataset.
  
Steps
* Detect the puzzle from the image and crop it out. 
* Find all 81 boxes and predict the numbers in them.
* CNN used for number prediction trained on MNIST Dataset.
* Create array from predictions
* Solve array using backtracking algorithm
* Print Numbers back into the image and save

# Input Image
![alt text](https://github.com/VedantDesai11/SudokuSolver/blob/master/SudokuPuzzles/Puzzle1.png)

# Output Image
![alt text](https://github.com/VedantDesai11/SudokuSolver/blob/master/SudokuPuzzles/Puzzle1Solution.png)
