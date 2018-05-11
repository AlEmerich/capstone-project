import argparse
import json
import threading
from walk.agents.ac_policy import AC_Policy

def launch_tensorboard(log_dir):
    import tensorflow as tf    
    from tensorboard import main as tb
    tf.flags.FLAGS.logdir = log_dir
    tb.main()

# t = threading.Thread(target=launch_tensorboard, args=(["plots/"]))
# t.start()

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


def load_param_from_json():
    return json.load(open("params.json"))


def load_param_from_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", type=str2bool, default=False,
                        help="t to reset the environment when done, f if not. (default: f)")
    parser.add_argument("--render", type=str2bool, default=False,
                        help="t to render the environment, f if not. (default: f)")
    parser.add_argument("--plot", type=str, default=None,
                        help="show library to plot metrics, matplotlib or tensorflow (default: None)")

    # Hyper parameters
    parser.add_argument("--train", type=str2bool, default=True,
                        help="True to perform training, false if not. (default: True)")
    parser.add_argument("--load_weights", type=str2bool, default=False,
                        help="True to load weights from file, false if not. (default: False)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of batch to get from memory at every step. (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of maximum loop before reset at each train_pass. (default: 1000)")

    parser.add_argument("--epsilon", type=float, default=0.8,
                        help="Epsilon value to make random decisions sometimes. (default: 0.8)")
    parser.add_argument("--epsilon_decay", type=float, default=0.001,
                        help="Decay to decrease epsilon. (default: 0.001)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="How fast the model will learn. (default: 0.01)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="How much target modes wil lbe updated from models. (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Discount factor. (default: 0.01)")
    return vars(parser.parse_args())


def main(args):
    """Entry point of the program.
    """
    # Send args to the actor and let it train
    humanoid = AC_Policy(args)
    humanoid.run()


if __name__ == "__main__":
    main(load_param_from_json())

