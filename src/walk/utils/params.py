class Params():
    """Hold every argument of the program.
    """

    def __init__(self, args):
        """Unpack args into memebrs."""
        self.render = args["render"]
        self.reset = args["reset"]
        self.plot = args["plot"]

        self.train = args["train"]
        self.load_weights = args["load_weights"]
        self.batch_size = args["batch_size"]

        self.epochs = args["epochs"]
        self.train_pass = args["train_pass"]

        self.base_epsilon = args["epsilon"]
        self.epsilon = self.base_epsilon
        self.epsilon_decay = args["epsilon_decay"]

        self.tau = args["tau"]
        self.learning_rate = args["learning_rate"]
        self.gamma = args["gamma"]
