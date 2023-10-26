import numpy as np
from solver import Solver
import sys, getopt
from queue import PriorityQueue, Queue
#from game_state import GameState
#import numpy as np
import time

class Solver():
    def __init__(self, init_state, goal_state, heuristic_func="manhattan", max_iter=2500):
        self.__init_state = init_state
        self.__goal_state = goal_state
        self.__heuristic_func = heuristic_func
        self.__MAX = 100000
        self.__max_iter = max_iter
        self.__path = []
        self.__number_of_steps = 0
        self.__summary = ""

    def set_max_iter(self, max_iter):
        self.__max_iter = max_iter

    def get_path(self):
        return self.__path

    def get_summary(self):
        return self.__summary

    def solve_a_star(self):
        x_axis = [1, 0, -1, 0]
        y_axis = [0, 1, 0, -1]

        level = 0
        visited_nodes = set()

        start_time = time.time()

        nodes = PriorityQueue(self.__MAX)
        init_node = GameState(self.__init_state.flatten().tolist(), self.__goal_state.flatten().tolist(), level,
                              parent=None, heuristic_func=self.__heuristic_func)
        nodes.put(init_node)

        epochs = 0
        while nodes.qsize() and epochs <= self.__max_iter:
            epochs += 1

            cur_node = nodes.get()
            cur_state = cur_node.get_state()

            if str(cur_state) in visited_nodes:
                continue
            visited_nodes.add(str(cur_state))

            if cur_state == self.__goal_state.flatten().tolist():
                self.__summary = str("A* took " + str(
                    cur_node.get_level()) + " steps to get from initial state to the desired goal, visited total of " + str(
                    epochs) + " nodes, and took around " + str(
                    np.round(time.time() - start_time, 4)) + " seconds to reach the desired solution.")
                while cur_node.get_parent():
                    self.__path.append(cur_node)
                    cur_node = cur_node.get_parent()
                break

            empty_tile = cur_state.index(0)
            i, j = empty_tile // self.__goal_state.shape[0], empty_tile % self.__goal_state.shape[0]

            cur_state = np.array(cur_state).reshape(self.__goal_state.shape[0], self.__goal_state.shape[0])
            for x, y in zip(x_axis, y_axis):
                new_state = np.array(cur_state)
                if i + x >= 0 and i + x < self.__goal_state.shape[0] and j + y >= 0 and j + y < self.__goal_state.shape[
                    0]:
                    new_state[i, j], new_state[i + x, j + y] = new_state[i + x, j + y], new_state[i, j]
                    game_state = GameState(new_state.flatten().tolist(), self.__goal_state.flatten().tolist(),
                                           cur_node.get_level() + 1, cur_node, self.__heuristic_func)
                    if str(game_state.get_state()) not in visited_nodes:
                        nodes.put(game_state)
        if epochs > self.__max_iter:
            print('This grid setting is not solvable')
        return self.__path

    def solve_bfs(self):
        x_axis = [1, 0, -1, 0]
        y_axis = [0, 1, 0, -1]

        level = 0
        visited_nodes = set()

        start_time = time.time()

        nodes = Queue(self.__MAX)
        init_node = GameState(self.__init_state.flatten().tolist(), self.__goal_state.flatten().tolist(), level,
                              parent=None, heuristic_func=self.__heuristic_func)
        nodes.put(init_node)

        epochs = 0
        while nodes.qsize() and epochs <= self.__max_iter:
            epochs += 1

            cur_node = nodes.get()
            cur_state = cur_node.get_state()

            if str(cur_state) in visited_nodes:
                continue
            visited_nodes.add(str(cur_state))

            if cur_state == self.__goal_state.flatten().tolist():
                self.__summary = str("BFS took " + str(
                    cur_node.get_level()) + " steps to get from initial state to the desired goal, visited total of " + str(
                    epochs) + " nodes, and took around " + str(
                    np.round(time.time() - start_time, 4)) + " seconds to reach the desired solution.")
                while cur_node.get_parent():
                    self.__path.append(cur_node)
                    cur_node = cur_node.get_parent()
                break

            empty_tile = cur_state.index(0)
            i, j = empty_tile // self.__goal_state.shape[0], empty_tile % self.__goal_state.shape[0]

            cur_state = np.array(cur_state).reshape(self.__goal_state.shape[0], self.__goal_state.shape[0])
            for x, y in zip(x_axis, y_axis):
                new_state = np.array(cur_state)
                if i + x >= 0 and i + x < self.__goal_state.shape[0] and j + y >= 0 and j + y < self.__goal_state.shape[
                    0]:
                    new_state[i, j], new_state[i + x, j + y] = new_state[i + x, j + y], new_state[i, j]
                    game_state = GameState(new_state.flatten().tolist(), self.__goal_state.flatten().tolist(),
                                           cur_node.get_level() + 1, cur_node, self.__heuristic_func)
                    if str(game_state.get_state()) not in visited_nodes:
                        nodes.put(game_state)
        if epochs > self.__max_iter:
            print('This grid setting is not solvable')
        return self.__path


class GameState():
    def __init__(self, state, goal_state, level, parent=None, heuristic_func="manhattan"):
        self.__state = state
        self.__goal_state = goal_state
        self.__level = level
        self.__heuristic_func = heuristic_func
        self.__heuristic_score = level
        self.__parent = parent
        self.calculate_fitness()

    def __hash__(self):
        return hash(str(self.__state))

    def __lt__(self, other):
        return self.__heuristic_score < other.__heuristic_score

    def __eq__(self, other):
        return self.__heuristic_score == other.__heuristic_score

    def __gt__(self, other):
        return self.__heuristic_score > other.__heuristic_score

    def get_state(self):
        return self.__state

    def get_score(self):
        return self.__heuristic_score

    def get_level(self):
        return self.__level

    def get_parent(self):
        return self.__parent

    def calculate_fitness(self):
        if self.__heuristic_func == "misplaced_tiles":
            for cur_tile, goal_tile in zip(self.__state, self.__goal_state):
                if cur_tile != goal_tile:
                    self.__heuristic_score += 1
        elif self.__heuristic_func == "manhattan":
            for cur_tile in self.__state:
                cur_idx = self.__state.index(cur_tile)
                goal_idx = self.__goal_state.index(cur_tile)
                cur_i, cur_j = cur_idx // int(np.sqrt(len(self.__state))), cur_idx % int(np.sqrt(len(self.__state)))
                goal_i, goal_j = goal_idx // int(np.sqrt(len(self.__state))), goal_idx % int(np.sqrt(len(self.__state)))
                self.__heuristic_score += self.calculate_manhattan(cur_i, cur_j, goal_i, goal_j)
        else:
            print('Unknown heuristic function is being used.')

    def calculate_manhattan(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)


# samples
# goal_state = np.array([[1, 2, 3],
#                       [4, 5, 6],
#                       [7, 8, 0]])
#
# init_state = np.array([[1, 8, 7],
#                       [3, 0, 5],
#                       [4, 6, 2]])

def BFS(init_state, goal_state, max_iter, heuristic):
    solver = Solver(init_state, goal_state, heuristic, max_iter)
    path = solver.solve_bfs()

    if len(path) == 0:
        exit(1)

    init_idx = init_state.flatten().tolist().index(0)
    init_i, init_j = init_idx // goal_state.shape[0], init_idx % goal_state.shape[0]

    print()
    print('INITIAL STATE')
    for i in range(goal_state.shape[0]):
        print(init_state[i, :])
    print()
    for node in reversed(path):
        cur_idx = node.get_state().index(0)
        cur_i, cur_j = cur_idx // goal_state.shape[0], cur_idx % goal_state.shape[0]

        new_i, new_j = cur_i - init_i, cur_j - init_j
        if new_j == 0 and new_i == -1:
            print('Moved UP    from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        elif new_j == 0 and new_i == 1:
            print('Moved DOWN  from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        elif new_i == 0 and new_j == 1:
            print('Moved RIGHT from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        else:
            print('Moved LEFT  from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        print('Score using ' + heuristic + ' heuristic is ' + str(
            node.get_score() - node.get_level()) + ' in level ' + str(node.get_level()))

        init_i, init_j = cur_i, cur_j

        for i in range(goal_state.shape[0]):
            print(np.array(node.get_state()).reshape(goal_state.shape[0], goal_state.shape[0])[i, :])
        print()
    print(solver.get_summary())


def A_star(init_state, goal_state, max_iter, heuristic):
    solver = Solver(init_state, goal_state, heuristic, max_iter)
    path = solver.solve_a_star()

    if len(path) == 0:
        exit(1)

    init_idx = init_state.flatten().tolist().index(0)
    init_i, init_j = init_idx // goal_state.shape[0], init_idx % goal_state.shape[0]

    print()
    print('INITIAL STATE')
    for i in range(goal_state.shape[0]):
        print(init_state[i, :])
    print()
    for node in reversed(path):
        cur_idx = node.get_state().index(0)
        cur_i, cur_j = cur_idx // goal_state.shape[0], cur_idx % goal_state.shape[0]

        new_i, new_j = cur_i - init_i, cur_j - init_j
        if new_j == 0 and new_i == -1:
            print('Moved UP    from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        elif new_j == 0 and new_i == 1:
            print('Moved DOWN  from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        elif new_i == 0 and new_j == 1:
            print('Moved RIGHT from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        else:
            print('Moved LEFT  from ' + str((init_i, init_j)) + ' --> ' + str((cur_i, cur_j)))
        print('Score using ' + heuristic + ' heuristic is ' + str(
            node.get_score() - node.get_level()) + ' in level ' + str(node.get_level()))

        init_i, init_j = cur_i, cur_j

        for i in range(goal_state.shape[0]):
            print(np.array(node.get_state()).reshape(goal_state.shape[0], goal_state.shape[0])[i, :])
        print()
    print(solver.get_summary())


def main(argv):
    max_iter = 5000     # >= 10000
    heuristic = "manhattan"     # or misplaced_tiles
    algorithm = "a_star"    # or bfs
    n = 3   # or n (generic)

    try:
        opts, args = getopt.getopt(argv, "hn:", ["mx=", "heur=", "astar", "bfs"])
    except getopt.GetoptError:
        print(
            'python sliding_puzzle.py -h <help> -n <matrix shape ex: n = 3 -> 3x3 matrix> --mx <maximum_nodes> --heur <heuristic> --astar (default algorithm) or --bfs')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'python sliding_puzzle.py -h <help> -n <matrix shape ex: n = 3 -> 3x3 matrix> --mx <maximum_nodes> --heur <heuristic> --astar (default algorithm) or --bfs')
            sys.exit()
        elif opt == '-n':
            n = int(arg)
        elif opt in ("--mx"):
            max_iter = int(arg)
        elif opt in ("--heur"):
            if arg == "manhattan" or arg == "misplaced_tiles":
                heuristic = (arg)
        elif opt in ("--astar"):
            algorithm = "a_star"
        elif opt in ("--bfs"):
            algorithm = "bfs"

    while True:
        try:
            init_state = input("Enter a list of " + str(
                n * n) + " numbers representing the inital state, SEPERATED by WHITE SPACE(1 2 3 etc.): ")
            init_state = init_state.split()
            for i in range(len(init_state)):
                init_state[i] = int(init_state[i])
            goal_state = input("Enter a list of " + str(
                n * n) + " numbers representing the goal state, SEPERATED by WHITE SPACE(1 2 3 etc.): ")
            goal_state = goal_state.split()
            for i in range(len(goal_state)):
                goal_state[i] = int(goal_state[i])
            if len(goal_state) == len(init_state) and len(goal_state) == n * n:
                break
            else:
                print("Please re-enter the input again correctly")
        except Exception as ex:
            print(ex)

    init_state = np.array(init_state).reshape(n, n)
    goal_state = np.array(goal_state).reshape(n, n)

    if algorithm == "a_star":
        A_star(init_state, goal_state, max_iter, heuristic)
    else:
        BFS(init_state, goal_state, max_iter, heuristic)


if __name__ == "__main__":
    main(sys.argv[1:])