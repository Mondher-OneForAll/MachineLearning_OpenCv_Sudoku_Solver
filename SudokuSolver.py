board = [[9, 0, 0, 0, 4, 3, 1, 6, 0],
         [0, 0, 0, 0, 0, 2, 0, 0, 0],
         [0, 0, 8, 0, 0, 0, 0, 9, 0],
         [8, 0, 0, 0, 1, 9, 3, 0, 0],
         [0, 5, 0, 0, 0, 0, 0, 0, 7],
         [0, 0, 0, 6, 0, 0, 0, 0, 0],
         [0, 0, 0, 8, 0, 0, 6, 0, 0],
         [0, 0, 7, 0, 6, 4, 0, 0, 3],
         [4, 0, 0, 2, 0, 0, 0, 0, 0]]


def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve(board):
                return True

            board[row][col] = 0

    return False


def valid(board, num, pos):
    # Check columns
    for j in range(len(board[pos[0]])):
        if board[pos[0]][j] == num and pos[1] != j:
            return False

            # Check rows
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

            # Check box
    box_x = pos[0] // 3
    box_y = pos[1] // 3

    for i in range(box_x * 3, box_x * 3 + 3):
        for j in range(box_y * 3, box_y * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True


def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("---------------------")

        for j in range(len(board[i])):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            print(board[i][j], end=" ")

        print("")


def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return i, j

    return False

# print_board(board)
# solve(board)
# print("-------Solved-------")
# print_board(board)
