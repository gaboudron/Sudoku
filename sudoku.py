import copy
import heapq
import time
from typing import List, Tuple

class Grid_State:
    """Represents a 9x9 Sudoku grid."""
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initializes the Sudoku grid state.
        
        Parameters:
            - grid (List[List[int]]): A 2D list representing the Sudoku grid.
        """
        self.grid = grid
        self.N_ROWS = len(self.grid)
        self.N_COLS = len(self.grid[0])
        self.current_position = self.get_position()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Sudoku grid.

        Returns:
            - str: The string representation of the grid.
        """
        return '\n'.join(' '.join(str(self.grid[row][col]) for col in range(self.N_COLS)) for row in range(self.N_ROWS)) + '\n'

    def get_position(self) -> Tuple[int, int]:
        """
        Finds the next empty position (cell with value 0) in the grid.
        
        Returns:
            - Tuple[int, int]: The row and column indices of the empty position, or None if the grid is complete.
        """
        for row in range(self.N_ROWS):
            for col in range(self.N_COLS):
                if self.grid[row][col] == 0:
                    return (row, col)
        return None

    def is_valid(self, position: Tuple[int, int], num: int) -> bool:
        """
        Checks if placing a number at a specific position is valid.
        
        Parameters:
            - position (Tuple[int, int]): The row and column indices where the number is to be placed.
            - num (int): The number to place in the grid.
        
        Returns:
            - bool: True if the placement is valid, False otherwise.
        """
        row, col = position

        if any(self.grid[row][j] == num for j in range(self.N_COLS)):  # Check row
            return False
        if any(self.grid[i][col] == num for i in range(self.N_ROWS)):  # Check column
            return False

        # Check 3x3 sub-grid
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.grid[i][j] == num:
                    return False
        return True

    def count_constraints(self) -> int:
        """
        Counts the number of valid placements for each empty cell in the grid.
        
        Returns:
            - int: The total count of valid placements across all empty cells.
        """
        count = 0
        for row in range(self.N_ROWS):
            for col in range(self.N_COLS):
                if self.grid[row][col] == 0:
                    count += sum(1 for num in range(1, 10) if self.is_valid((row, col), num))
        return count

    def __hash__(self):
        """
        Computes a hash for the grid state to enable its use in sets or as dictionary keys.
        
        Returns:
            - int: The hash value of the grid state.
        """
        return hash(tuple(map(tuple, self.grid)))

    def __eq__(self, other):
        """
        Checks equality between two grid states.
        
        Parameters:
            - other (Grid_State): Another grid state to compare with.
        
        Returns:
            - bool: True if the grids are equal, False otherwise.
        """
        return self.grid == other.grid


class Sudoku_Problem:
    """Defines the Sudoku problem."""

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the Sudoku problem with the initial grid.
        
        Parameters:
            - grid (List[List[int]]): A 2D list representing the Sudoku grid.
        """
        self.initial_grid = grid

    def initial_state(self) -> Grid_State:
        """
        Returns the initial state of the Sudoku problem.
        
        Returns:
            - Grid_State: The initial grid state.
        """
        return Grid_State(self.initial_grid)

    def actions(self, state: Grid_State) -> List[int]:
        """
        Returns valid numbers for the current empty position in the grid.
        
        Parameters:
            - state (Grid_State): The current grid state.
        
        Returns:
            - List[int]: A list of valid numbers that can be placed at the current position.
        """
        return [num for num in range(1, 10) if state.is_valid(state.current_position, num)] if state.current_position else []

    def succ(self, state: Grid_State, action: int) -> Grid_State:
        """
        Generates a new state by placing a number at the current empty position.
        
        Parameters:
            - state (Grid_State): The current grid state.
            - action (int): The number to place in the current position.
        
        Returns:
            - Grid_State: A new grid state with the number placed.
        """
        new_grid = copy.deepcopy(state.grid)
        row, col = state.current_position
        new_grid[row][col] = action
        return Grid_State(new_grid)

    def goal_test(self, state: Grid_State) -> bool:
        """
        Checks if the current state is a goal state (no empty positions).
        
        Parameters:
            - state (Grid_State): The current grid state.
        
        Returns:
            - bool: True if the grid has no empty positions, False otherwise.
        """
        return state.get_position() is None

    def heuristic(self, state: Grid_State) -> int:
        """
        Estimates the number of empty cells remaining in the grid.
        
        Parameters:
            - state (Grid_State): The current grid state.
        
        Returns:
            - int: The number of empty cells in the grid.
        """
        return sum(row.count(0) for row in state.grid)


class Node:
    """Represents a node in the search tree."""
    
    def __init__(self, state: Grid_State, parent=None, action=None, path_cost=0, heuristic=0):
        """
        Initializes a node with the given state and other parameters.
        
        Parameters:
            - state (Grid_State): The grid state represented by this node.
            - parent (Node): The parent node in the search tree.
            - action (int): The action that led to this state.
            - path_cost (int): The cost of the path from the root to this node.
            - heuristic (int): The heuristic value of this node.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.f_value = path_cost + heuristic

    def __lt__(self, other):
        """
        Compares two nodes based on their f-values for priority queue operations.
        
        Parameters:
            - other (Node): Another node to compare with.
        
        Returns:
            - bool: True if this node's f-value is less than the other's, False otherwise.
        """
        return self.f_value < other.f_value

    def expand(self, problem: Sudoku_Problem) -> List["Node"]:
        """
        Generates child nodes by applying valid actions to the current state.
        
        Parameters:
            - problem (Sudoku_Problem): The Sudoku problem instance.
        
        Returns:
            - List[Node]: A list of child nodes generated from the current node.
        """
        return [
            Node(problem.succ(self.state, action), self, action, self.path_cost + 1, 0)
            for action in problem.actions(self.state)
        ]

    def path(self) -> List["Node"]:
        """
        Generates the path from the root to this node.

        Returns:
            - List[Node]: A list of nodes, from the root to the node.
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


# ______________________________________________________________________________
# Search algorithms

def breadth_first_tree_search(problem: Sudoku_Problem):
    """
    Performs Breadth-First Search (BFS).
    
    Parameters:
        - problem (Sudoku_Problem): The Sudoku problem instance.

    Returns:
        if there is a solution:
            - node: The final state of a Sudoku grid.
        else:
            - None.
    """
    start_time = time.time()  # Start timer
    frontier = [Node(problem.initial_state())]
    nodes_explored = 0
    explored = set()

    while frontier:
        node = frontier.pop(0)  # FIFO
        nodes_explored += 1

        if problem.goal_test(node.state):
            print(node.state)
            print(f"Time: {time.time() - start_time:.4f}s")
            print(f"Nodes explored: {nodes_explored}")
            return node

        explored.add(tuple(map(tuple, node.state.grid)))

        for child in node.expand(problem):
            if tuple(map(tuple, child.state.grid)) not in explored:
                frontier.append(child)

    print('No solution found.')
    return None


def depth_first_tree_search(problem: Sudoku_Problem):
    """
    Performs Depth-First Search (DFS).
        
    Parameters:
        - problem (Sudoku_Problem): The Sudoku problem instance.

    Returns:
        if there is a solution:
            - node: The final state of a Sudoku grid.
        else:
            - None.
    """
    start_time = time.time()  # Start timer
    frontier = [Node(problem.initial_state())]
    nodes_explored = 0
    explored = set()

    while frontier:
        node = frontier.pop()  # LIFO
        nodes_explored += 1

        if problem.goal_test(node.state):
            print(node.state)
            print(f"Time: {time.time() - start_time:.4f}s")
            print(f"Nodes explored: {nodes_explored}")
            return node

        explored.add(tuple(map(tuple, node.state.grid)))

        for child in node.expand(problem):
            if tuple(map(tuple, child.state.grid)) not in explored:
                frontier.append(child)

    print('No solution found.')
    return None


def a_star_search(problem: Sudoku_Problem):
    """
    Performs A* Search.
        
    Parameters:
        - problem (Sudoku_Problem): The Sudoku problem instance.

    Returns:
        if there is a solution:
            - node: The final state of a Sudoku grid.
        else:
            - None.
    """
    start_time = time.time()  # Start timer
    frontier = []  # (f_value, node) pairs
    initial_node = Node(problem.initial_state(), heuristic=problem.heuristic(problem.initial_state()))
    nodes_explored = 0
    heapq.heappush(frontier, (initial_node.f_value, initial_node))
    explored = set()

    while frontier:
        _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            print(node.state)
            print(f"Time: {time.time() - start_time:.4f}s")
            print(f"Nodes explored: {nodes_explored}")
            return node

        explored.add(tuple(map(tuple, node.state.grid)))

        for child in node.expand(problem):
            if tuple(map(tuple, child.state.grid)) not in explored:
                child.heuristic = problem.heuristic(child.state)
                child.f_value = child.path_cost + child.heuristic
                heapq.heappush(frontier, (child.f_value, child))

    print('No solution found.')
    return None


def greedy_best_first_search(problem: Sudoku_Problem):
    """
    Performs Greedy Best-First Search.
        
    Parameters:
        - problem (Sudoku_Problem): The Sudoku problem instance.

    Returns:
        if there is a solution:
            - node: The final state of a Sudoku grid.
        else:
            - None.
    """
    start_time = time.time()  # Start timer
    frontier = []
    initial_node = Node(problem.initial_state(), heuristic=problem.heuristic(problem.initial_state()))
    frontier.append(initial_node)
    nodes_explored = 0
    explored = set()

    while frontier:
        frontier.sort(key=lambda node: node.heuristic)  # Sort by heuristic
        node = frontier.pop(0)
        nodes_explored += 1

        if problem.goal_test(node.state):
            print(node.state)
            print(f"Time: {time.time() - start_time:.4f}s")
            print(f"Nodes explored: {nodes_explored}")
            return node

        explored.add(tuple(map(tuple, node.state.grid)))

        for child in node.expand(problem):
            if tuple(map(tuple, child.state.grid)) not in explored:
                child.heuristic = problem.heuristic(child.state)
                frontier.append(child)

    print('No solution found.')
    return None


def solve_sudoku(problem, algo):
    """
    Function to choose which search algorithm you to solve the problem

    Parameters:
        - problem (Sudoku_Problem): the Sudoku.
        - algo (String): the name of the search algorithm.

    Returns:
        - solution_node (Node): The final state of the grid.
    """
    if algo == "DFS":
        solution_node = depth_first_tree_search(problem)
    if algo == "BFS":
        solution_node = breadth_first_tree_search(problem)
    if algo == "A*":
        solution_node = a_star_search(problem)
    if algo == "Greedy":
        solution_node = greedy_best_first_search(problem)
    return solution_node

# ______________________________________________________________________________
# Solve

# A grid with 20 cases
grid_20 = [
    [2, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 7, 0, 9, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 9, 0],
    [9, 0, 0, 0, 5, 0, 0, 0, 2],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 2, 0],
    [4, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 5, 0, 2, 0, 0, 0, 0, 7],
]

# A grid with 30 cases
grid_30 = [
    [2, 0, 3, 0, 0, 9, 0, 7, 0],
    [0, 9, 0, 0, 0, 0, 2, 6, 0],
    [8, 0, 0, 0, 7, 0, 9, 0, 0],
    [0, 2, 5, 0, 6, 0, 0, 9, 0],
    [9, 0, 0, 0, 5, 0, 0, 0, 2],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 4, 0, 0, 6, 2, 0],
    [4, 0, 0, 0, 9, 6, 0, 1, 0],
    [0, 5, 0, 2, 0, 0, 0, 0, 7],
]

sudoku_problem = Sudoku_Problem(grid_20)

# Choose the search algorithm
solution_node = solve_sudoku(sudoku_problem, "Greedy") # DFS, BFS, A*
