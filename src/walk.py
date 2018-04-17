import argparse
from walk.actors.random_policy import RandomPolicy

def str2bool(v):
    """Function to convert string respresenting a boolean
    into a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Entry point of the program
    """

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", type=str2bool, default=False)
    parser.add_argument("--render", type=str2bool, default=False)
    args = parser.parse_args()
    # Send args to the actor and let it train
    humanoid = RandomPolicy(args)
    humanoid.train()


if __name__ == "__main__":
    main()
