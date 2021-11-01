"""hw3-2.py
~~~~~~~~~~~~~~
The Classification of Chern/normal insulator.
It takes a lot of time to generate the datasets, if you do not have enough computing resources, you can use the dataset we provide.
Please run the program, tune parameters and plot the phase diagrams.
Submit the runtime outputs and the best phase diagram on the course.pku.edu.cn.
"""

#### Libraries
# Standard library
import os
import json
import random
import sys
import math

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(0)
# random.seed(0)


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


#### Main Network class
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.
    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(x):
    x_exp = np.exp(x)
    x_exp_row_sum = np.tile(np.sum(x_exp, axis=0), (2, 1))
    return x_exp / x_exp_row_sum

def load_data_wrapper(tr_d, tr_r, te_d, te_r):
    training_data = list(zip(tr_d, [vectorized_result(y) for y in tr_r]))
    test_data = list(zip(te_d, te_r))
    return (training_data, test_data)

def plot_fig(x, p0, p1, xlabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, p0, '-b', label='p0')
    ax2 = ax.twinx()
    ax2.plot(x, p1, '-r', label='p1')
    ax.legend(loc=6)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("p0")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("p1")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc=7)
    plt.title(title)
    plt.savefig("{}.png".format(title))


"""
Chern/normal insulator sample generator
~~~~~~~~~~~~
"""

def generate_insulator_data(count, qlt, kappa=1.0, delta=0.25, dmax=1, L=12, autocorr=4*12*12):
    """Generate a specific number of samples of the Chern/normal model with model parameter delta = 0.5 and kappa.
    The default kappa is at the critical kappa_c. dmax controls the cut-off length scale of the operators. qlt is a list of local triangles."""
    N = int(L*L/2)
    training_data = []
    conf = -np.ones((L, L), dtype=int)
    index = np.zeros((L, L), dtype=int)
    corr=np.zeros((L,L,2*dmax+1,2*dmax+1,3),dtype=complex)
    sample = np.zeros((L,L,len(qlt)),dtype=float)
    for x in range(L):
        for y in range(L):
            if (x-y)%2 == 0:
                conf[x,y]*=-1
            index[x,y] = int((y*L+x)/2)
            """ Initial configuration. Checkerboard occupation of electrons, 1 for occupied and -1 for unoccupied.
            Index keep track the row number of the (x, y) fermion in the Slater determinant (and its supplement). """
    SD = np.zeros((N, N), dtype=complex)
    SD2= np.zeros((N, N), dtype=complex)
    """ Initialize the Slater determinant (for the occupied sites) and its supplement (for the unoccupied sites).
    Changing the configuration will switch the corresponding rows of the Slater determinant and its supplement."""
    for kx in range(L):
        for ky in range(int(L/2)):
            energy = math.sqrt(math.cos(kx*2*np.pi/L)**2 + math.cos(ky*2*np.pi/L)**2 + (2*delta*math.sin(ky*2*np.pi/L)*(1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
            for x in range(L):
                for y in range(L):
                    if conf[x,y] == 1:
                        """Occupied site, go to Slater determinant."""
                        if (y%2 == 0):
                            """First sublattice."""
                            SD[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) *  \
                            complex(math.cos(ky*2*np.pi/L), -2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L))) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                        else:
                            """Second sublattice."""
                            SD[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) * \
                            (math.cos(kx*2*np.pi/L)+energy) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                    else:
                        """Unoccupied site, go to supplement."""
                        if (y%2 == 0):
                            """First sublattice."""
                            SD2[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) *  \
                            complex(math.cos(ky*2*np.pi/L), -2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L))) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                        else:
                            """Second sublattice."""
                            SD2[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) * \
                            (math.cos(kx*2*np.pi/L)+energy) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
    SDinv = np.linalg.inv(SD)
    #print(np.amax(abs(np.dot(SD, SDinv)-np.diag(np.ones(N)))))
    """Also keep the inverse of Slater determinant. Initialization complete."""
    for icount in range(count*30+10):
        for iautocorr in range(autocorr):
            """ A number of steps between measurements to ensure (approximately) independent samples."""
            while True:
                x, y = np.random.randint(L, size=2)
                if conf[x,y] == 1:
                    break
            while True:
                xp, yp = np.random.randint(L, size=2)
                if conf[xp,yp] == -1:
                    break
            if (np.random.rand()<abs(1+np.dot(SD2[index[xp,yp],:]-SD[index[x,y],:],SDinv[:, index[x,y]]))**2):
                """Calculate the update probability. Accept and update configuration."""
                conf[x,y] *=-1
                conf[xp,yp] *=-1
                """Update matrix inverse and matrices."""
                SDinv[:,index[x,y]] /= np.dot(SD2[index[xp,yp],:],SDinv[:,index[x,y]])
                tempnp = np.reshape(np.dot(SD2[index[xp,yp],:], SDinv),N)
                for i in range(N):
                    if i != index[x,y]:
                        SDinv[:,i]-=SDinv[:,index[x,y]]*tempnp[i]
                tempnp = SD[index[x,y],:]
                SD[index[x,y],:] = SD2[index[xp,yp],:]
                SD2[index[xp,yp],:] = tempnp
                tempi = index[x,y]
                index[x,y] = index[xp,yp]
                index[xp,yp] = tempi
        if(icount > 9):
            for x in range(L):
                for y in range(L):
                    for dx in range(2*dmax+1):
                        for dy in range(2*dmax+1):
                            xp = modpbc(x+dx-dmax,L)
                            yp = modpbc(y+dy-dmax,L)
                            if(conf[x,y]==1 and conf[xp,yp]==-1):
                                corr[x,y,dx,dy,(icount-10)%3] = 1+np.dot(SD2[index[xp,yp],:]-SD[index[x,y],:],SDinv[:,index[x,y]])
                            else:
                                corr[x,y,dx,dy,(icount-10)%3] = 0
            if ((icount-10)%3 == 2):
                for x in range(L):
                    for y in range(L):
                        for iqlt in range(len(qlt)):
                            xp = modpbc(x+qlt[iqlt][0],L)
                            yp = modpbc(y+qlt[iqlt][1],L)
                            xpp = modpbc(xp+qlt[iqlt][2],L)
                            ypp = modpbc(yp+qlt[iqlt][3],L)
                            sample[x,y,iqlt]+=(corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,0] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,1]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 2]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,1] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,2]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 0]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,2] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,0]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 1]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,0] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,2]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 1]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,2] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,1]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 0]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,1] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,0]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 2]).imag
            if ((icount-10)%30 == 29):
                training_data.append(np.reshape(sample, (L**2*len(qlt), 1)))
                sample = np.zeros((L,L,len(qlt)),dtype=float)
                #print(np.amax(abs(np.dot(SD, SDinv)-np.diag(np.ones(N)))))
    return training_data

def modpbc(x, L=12):
    """Ensure the coordinate is between [0, L-1] by applying periodic boundary condition. """
    if (x<0):
        return modpbc(x+L, L)
    elif (x>L-1):
        return modpbc(x-L, L)
    else:
        return x

def countriangle(dmax=1):
    """The list of triangles within a cut-off scale dmax."""
    qlt = []
    for dx in range(-dmax, dmax+1):
        for dy in range(dmax+1):
            for dx2 in range(-dmax, dmax+1):
                for dy2 in range(-dy, dmax+1):
                    if(abs(dx+dx2) <= dmax and abs(dy+dy2) <= dmax and (dy2*dx-dy*dx2)>0 and (dy>0 or dx>0) and (dx+dx2>0 or dy+dy2>0)):
                        qlt.append([dx, dy, dx2, dy2])
    return qlt


if __name__ == "__main__":
    train = True

    # Chern/normal insulator: Kappa < 0.5: label 0, Kappa > 0.5: label 1
    print("Chern/normal insulator: Kappa < 0.5: label 0, Kappa > 0.5: label 1")
    # load or generate dataset
    if os.path.exists("Insulator_dataset.npy"):
        training_data, test_data = np.load("Insulator_dataset.npy", allow_pickle=True)
        print("Datasets loading finished!")
    else:
        train_count, test_count = 10000, 1000
        tr_d, tr_r, te_d, te_r = [], [], [], []
        tr_d += generate_insulator_data(int(train_count / 2), countriangle(1), kappa=0.1)
        tr_r += int(train_count / 2) * [0]
        tr_d += generate_insulator_data(int(train_count / 2), countriangle(1), kappa=1.0)
        tr_r += int(train_count / 2) * [1]
        te_d += generate_insulator_data(int(test_count / 2), countriangle(1), kappa=0.1)
        te_r += int(test_count / 2) * [0]
        te_d += generate_insulator_data(int(test_count / 2), countriangle(1), kappa=1.0)
        te_r += int(test_count / 2) * [1]
        training_data, test_data = load_data_wrapper(tr_d, tr_r, te_d, te_r)
        np.save("Insulator_dataset.npy", (training_data, test_data))
        print("Datasets generation finished!")
    print("len(training_data)={},\tlen(test_data)={}".format(len(training_data), len(test_data)))
    # train & test
    if os.path.exists("Insulator_ANN.pkl") and not train:
        net = load("Insulator_ANN.pkl")
        print("ANN loading finished!\n\n")
    else:
        net = Network([training_data[0][0].shape[0], 30, 2])
        # net.large_weight_initializer()
        net.SGD(training_data, 100, 10, 0.1, lmbda=5.0, evaluation_data=test_data, monitor_evaluation_accuracy=True,
               monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
        net.save('Insulator_ANN.pkl')
        print("ANN training finished!\n\n")
    # plot phase diagram
    Kap, p0, p1, group = np.arange(0.1, 1.01, 0.03), [], [], 10
    for k in Kap:
        samples = np.concatenate(generate_insulator_data(group, countriangle(1), kappa=k), axis=1)
        output = net.feedforward(samples)
        p0.append(np.nanmean(output[0]))
        p1.append(np.nanmean(output[1]))
    plot_fig(Kap, p0, p1, "Kappa", "Insulator")
