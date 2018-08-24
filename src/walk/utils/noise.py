import numpy as np

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


class Noise:
    """Implements the Ornstein-Uhlenbeck noise to add to the output
    of the policy network.
    """
    def __init__(self, mu, sigma=0.1, theta=.15, dt=1e-2, x0=None):
        """Initiliaze parameters mu, sigma, theta, dt and x0."""
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """Compute *x-1* + theta *(mu - *x-1*) * dt + sigma * sqrt(dt) * normal(shape(mu))
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(
                size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """reset *x-1* value."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
