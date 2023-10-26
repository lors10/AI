import random

real_number = ''
real_heuristic = ''
real_algorithm = ''
real_random_range = ''

class RandomAgent:
    def generate_random_string(self):
        # Genera un numero casuale tra 0 e 10000
        random_number = random.randint(0, 10000)
        real_number = random_number

        # Scegli una stringa casuale tra 'manhattan' e 'misplaced_tiles'
        heuristic = random.choice(['manhattan', 'misplaced_tiles'])
        real_heuristic = heuristic

        # Scegli una stringa casuale tra 'astar' e 'bfs'
        algorithm = random.choice(['astar', 'bfs'])
        real_algorithm = algorithm

        # Genera un numero casuale tra 2 e 4
        random_range = random.randint(2, 4)
        real_random_range = random_range

        # Componi la stringa con i valori generati
        random_string = f"{random_number} {heuristic} {algorithm} {random_range}"

        return random_string + real_number + real_heuristic + real_algorithm + real_random_range

def f_exampe():
    #   max_iter =  5000   >= 10000
    #   heuristic = "manhattan"      or misplaced_tiles
    #   algorithm = "a_star"     or bfs
    #   n = 3    or n (generic)

    max_iter = real_number
    print(max_iter)
    heuristic = real_heuristic
    print(heuristic)
    algorithm = real_algorithm
    print(algorithm)
    n = real_random_range
    print(n)

    #print(max_iter + heuristic + algorithm + n)
    f_exampe_string = f"{max_iter} {heuristic} {algorithm} {n}"

    return f_exampe_string

# Esempio di utilizzo
agent = RandomAgent()
random_string = agent.generate_random_string()
random_example_string = f_exampe()
print(random_string)
print(random_example_string)


