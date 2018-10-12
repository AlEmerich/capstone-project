from collections import namedtuple

class Params():
    """Hold every argument of the program.
    """

    def __init__(self, args):
        """Unpack args into members."""
        self.render = args["render"]
        self.reset = args["reset"]
        self.plot = args["plot"]

        self.train = args["train"]
        self.noisy = args["noisy"]
        self.load_weights = args["load_weights"]
        self.batch_size = args["batch_size"]

        self.epochs = args["epochs"]

        self.base_epsilon = args["epsilon"]
        self.epsilon = self.base_epsilon
        self.epsilon_decay = args["epsilon_decay"]

        self.tau = args["tau"]
        self.actor_learning_rate = args["actor_learning_rate"]
        self.critic_learning_rate = args["critic_learning_rate"]
        self.actor_batch_norm = args["actor_batch_norm"]
        self.critic_batch_norm = args["critic_batch_norm"]
        self.gamma = args["gamma"]

        self.norm_range = (args["norm_range"]["min"],
                           args["norm_range"]["max"])
        self.reward_multiply = args["reward_multiply"]
        self.dropout = args["dropout"]

        self.desc = args.__str__()

    def __str__(self):
        return self.desc
