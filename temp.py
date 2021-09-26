import random
import sys

def get_random_order(names):
    random.shuffle(names)

    print(' '.join(names))

if __name__ == "__main__":
    args = sys.argv[1:]
    get_random_order(args)