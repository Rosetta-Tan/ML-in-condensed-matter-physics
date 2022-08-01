import numpy as np
from collections import deque


class WolffMonteCarlo:
    """ Wolff Monte Carlo simulator for the Ising model."""

    def __init__(self, L, T, method=None):
        self._L = L
        self._T = T
        self._K = 2. / T
        self._method = method

        # set up initial state
        self._state = np.random.randint(0, 2, size=(L, L))

    @property
    def L(self):
        return self._L

    @property
    def T(self):
        return self._T

    @property
    def K(self):
        return self._K

    @property
    def state(self):
        return self._state

    def probability_add_bond(self, x1, y1, x2, y2, state):
        """The probability for adding a bond."""
        E = 1.0 if state[x1, y1] == state[x2, y2] else 0.0
        return 1.0 - np.exp(-self.K * E)

    def set_T(self, T):
        self._T = T
        self._K = 2. / T

    def wolff_iterative(self, state):
        """ Iterative Wolff Algorithm 
            This algorithm uses a doubly ended queue (deque), which provides O(1) 
            operations for adding and removing items from both ends of a list. 
        """

        # Convenient lists for indexing
        # Below, we'll use these to get the left (or 'above', etc) neighbors of a site.
        # Includes periodic boundaries! Another option would be to replace
        #    left[x1]   by   (x1 - 1) % L
        # but that does an addition and a module every step.
        left = [self.L - 1] + list(range(self.L - 1))
        right = list(range(1, self.L)) + [0]

        # Book-keeping containers
        sites_to_consider = deque()
        sites_to_flip = set()
        bonds_considered = set()

        # Initial queue of sites to consider, just consisting of a single (x,y) location
        sites_to_consider.append((
            np.random.randint(0, self.L),
            np.random.randint(0, self.L)
        ))

        # As long as there are sites to consider
        while sites_to_consider:
            # Pick a new site to consider from the queue, either using
            # breadth first or depth first
            if self._method == "BFS":
                x1, y1 = sites_to_consider.popleft()
            if self._method == "DFS":
                x1, y1 = sites_to_consider.pop()

            # For the neighbors of this site
            for x2, y2 in zip([left[x1], right[x1], x1, x1],
                              [y1, y1, left[y1], right[y1]]):

                # Check if we have not already considered this pair
                if not (x1, y1, x2, y2) in bonds_considered:
                    # Add the pair so that we don't flip it twice
                    bonds_considered.add((x1, y1, x2, y2))
                    bonds_considered.add((x2, y2, x1, y1))

                    if np.random.rand() < self.probability_add_bond(x1, y1, x2, y2, state):
                        sites_to_consider.append((x2, y2))
                        sites_to_flip.add((x1, y1))
                        sites_to_flip.add((x2, y2))

        return sites_to_flip

    def step(self):
        """Use Wolff and perform update."""

        # Get a list of sites to flip...
        to_flip = self.wolff_iterative(self._state)

        # ...and flip them
        for (x, y) in to_flip:
            self._state[x, y] = 1 - self._state[x, y]

        # Return the list of the flipped sites
        return to_flip


def generate_Ising_configurations(L, numSamplesPerT, Ts, equilibrationSteps=100):
    ''' Generates snapshots for the 2D Ising model for a given set of temperatures

    Samples are generated using Wolff cluster updates.

    Parameters:

        * `L`: Linear size of the system
        * `numSamplesPerT`: Number of samples to generate per temperature
        * `Ts`: List of temperatures
        * `equilibrationSteps`: Number of equilibration steps

    Returns:
        * A dictionary with the sampled configurations for each temperature

    '''

    # Initialize a new simulator
    sim = WolffMonteCarlo(L=L, T=5, method="DFS")

    all_data = {}

    # Loop over a fixed set of temperatures
    for T in Ts:
        print(f"Generating samples for L = {L:d} at T = {T:.3f}")

        # Set temperature
        sim.set_T(T)

        # For storing all of the configurations
        snapshots = []
        for s in range(numSamplesPerT + equilibrationSteps):

            # Keep flipping sites, until we flipped at least L^2 of them
            c = 0
            while c < 1:
                to_flip = sim.step()
                c = c + len(to_flip) / L / L

            # The first half of the flips are to equilibrate, the rest are samples
            if s >= equilibrationSteps:
                snapshots.append(np.array(-1 + 2 * sim.state.reshape(-1)))

        all_data[f'{T:.3f}'] = np.array(snapshots)

    return all_data


def split_training_data(all_data, Ts, Tc=2. / np.log(1 + np.sqrt(2)), train_fraction=0.8):
    # Lists to store the raw data
    raw_T = []
    raw_x = []
    raw_y = []

    for T in Ts:
        raw_x.append(all_data['%.3f' % (T)])
        n = len(all_data['%.3f' % (T)])
        label = [1, 0] if T < Tc else [0, 1]
        raw_y.append(np.array([label] * n))
        raw_T.append(np.array([T]*n))

    raw_T = np.concatenate(raw_T)
    raw_x = np.concatenate(raw_x, axis=0)
    raw_y = np.concatenate(raw_y, axis=0)

    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(raw_x))
    all_T = raw_T[indices]
    all_x = raw_x[indices]
    all_y = raw_y[indices]

    # Split into train and test sets
    train_split = int(train_fraction * len(all_x))
    train_T = np.asarray(all_T[:train_split])
    train_x = np.asarray(all_x[:train_split])
    train_y = np.asarray(all_y[:train_split])
    test_T = np.asarray(all_T[train_split:])
    test_x = np.asarray(all_x[train_split:])
    test_y = np.asarray(all_y[train_split:])

    return [raw_T, raw_x, raw_y], [train_T, train_x, train_y], [test_T, test_x, test_y]


if __name__ == '__main__':
    # The temperatures that we are going to generate samples at
    Ts = np.arange(1.95, 0.04, -0.1) * 2.27
    # For a few different system sizes, store the data in a dictionary with L as key
    all_data = generate_Ising_configurations(10, 1000, Ts)
    # train_Ts = list(Ts[:4]) + list(Ts[-4:])
    [raw_T, raw_x, raw_y], [train_T, train_x, train_y], [
        test_T, test_x, test_y] = split_training_data(all_data, Ts)
    np.savez('raw_dataset', T=raw_T, x=raw_x, y=raw_y)
    np.savez('train_dataset', T=train_T, x=train_x, y=train_y)
    np.savez('test_dataset', T=test_T, x=test_x, y=test_y)
