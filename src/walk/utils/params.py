class Params():
    """Hold every argument of the program.
    """

    def __init__(self, args):
        """Unpack args into memebrs."""
        self.render = args.render
        self.reset = args.reset
        self.plot = args.plot

        self.epochs = args.epochs
        self.train_pass = args.train_pass

        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
