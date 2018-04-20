import argparse
from walk.agents.random_policy import RandomPolicy

def str2bool(v):
    """Function to convert string respresenting a boolean
    into a boolean.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Entry point of the program.
    """

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", type=str2bool, default=False,
                        help="t to reset the environment when done, f if not. (default: f)")
    parser.add_argument("--render", type=str2bool, default=False,
                        help="t to render the environment, f if not. (default: f)")
    parser.add_argument("--plot", type=str2bool, default=True,
                        help="t to plot metrics, f if not (default: t)")

    # Hyper parameters
    parser.add_argument("--train_pass", type=int, default=10000,
                        help="Number of allowing reset, infinity if reset is false. (default: 100)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of maximum loop before reset at each train_pass. (default: 1000)")

    parser.add_argument("--epsilon", type=float, default=0.8,
                        help="Epsilon value to make random decisions sometimes. (default: 0.8)")
    parser.add_argument("--epsilon_decay", type=float, default=0.001,
                        help="Decay to decrease epsilon. (default: 0.001)")

    args = parser.parse_args()
    # Send args to the actor and let it train
    humanoid = RandomPolicy(args)
    humanoid.run(humanoid.params.train_pass, humanoid.params.epochs)


if __name__ == "__main__":
    main()
