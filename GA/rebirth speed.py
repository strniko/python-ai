import random
import string
import multiprocessing
from math import pow


def generate_random_string(length):
    letters = string.ascii_letters + " "  # includes both upper and lower case letters and space
    return ''.join(random.choice(letters) for _ in range(length))


def alter_string(input_string, alteration_rate):
    num_alterations = int(len(input_string) * alteration_rate)
    indices_to_alter = random.sample(range(len(input_string)), num_alterations)
    altered_string = list(input_string)
    for idx in indices_to_alter:
        altered_string[idx] = random.choice(string.ascii_letters + " ")
    return ''.join(altered_string)


def levenshtein_distance(s1, s2):
    dist = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            dist += 1
    return dist


def evaluate_child(child_string, goal):
    dist = levenshtein_distance(child_string, goal)
    return child_string, dist


def main():
    goal = input("Enter the Goal string here\n [+] ")
    print("")
    children = 200
    generations = 1000
    randomness = 0.9
    GOAT_str = generate_random_string(len(goal))
    GOAT_dist = len(goal) + 1

    with multiprocessing.Pool() as pool:
        for generation in range(generations):
            child_strings = [alter_string(GOAT_str, randomness) for _ in range(children)]
            results = pool.starmap(evaluate_child, [(child, goal) for child in child_strings])
            best_str, best_dist = min(results, key=lambda x: x[1])
            randomness = max(pow(0.9, 0.625 * generation), 0.1)
            print(f"The best result in generation {generation} was ➡ {best_str} ⬅ with a distance of {best_dist}.")
            if best_dist == 0:
                exit(0)
            else:
                print(f"The next rate is {randomness}")
                if best_dist < GOAT_dist:
                    GOAT_str = best_str
                    GOAT_dist = best_dist

    print(f"Your string was not found but the closest we got was {GOAT_str} with a distance of {GOAT_dist}.")
    exit(GOAT_dist)


if __name__ == '__main__':
    main()
