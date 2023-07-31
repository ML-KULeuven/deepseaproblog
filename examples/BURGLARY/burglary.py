import pickle

from examples.BURGLARY.architectures import MNIST_CNN, MNIST_Net, JointParameterNet
from examples.BURGLARY.data import create_dataloaders
from model import Model
from engines.tensor_ops import *
from network import Network
from examples.BURGLARY.evaluate import *
from problog.logic import Term, Constant
from query import Query


seed = 7
epochs = 2
lr = 1e-3

conv = MNIST_CNN()
network = MNIST_Net(conv, N=3)
network2 = MNIST_Net(conv, N=2)
mean1, mean2 = tf.random.uniform([2], 0, 10, seed=seed).numpy()
means = [mean1, mean2]
variance1, variance2 = tf.random.uniform([2], 2, 10, seed=seed)
variances = [variance1, variance2]

print('Initial spatial distribution values:')
print(mean1, float(variance1), mean2, float(variance2))

params = JointParameterNet(means, variances)
smallerthan = SmallerThan(maxmult=24., scheme="constant", alpha=1e-3)

nets = []
nets.append(Network(network, "earthq_net"))
nets.append(Network(network2, "burgl_net"))
nets.append(Network(params, "params"))
nets.append(Network(smallerthan, "smaller_than"))

model = Model("examples/BURGLARY/models/burglary.pl", nets, nb_samples=50)

# Set model loss and optimiser
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
model.set_loss(loss)
model.set_optimizer(optimiser)

# Load data
D = create_dataloaders("train", [[0, 1, 2], [8, 9]], simple=False, continuous_hearing=True)
D_val = create_dataloaders("val", [[0, 1, 2], [8, 9]], simple=False, continuous_hearing=True)[:100]

# Query and train
calls_input = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]), tf.keras.Input([1])]
model.compile_query("calls", calls_input)
model.train(D, epochs, val_data=D_val,
            eval_fns=[means_and_variances, evaluate_class_network, evaluate_class_network],
            fn_args=[[params, 10], [network, [0, 1, 2], 'val'], [network2, [8, 9], 'val', True, 10]],
            log_its=20)

acc1 = evaluate_class_network(network, [0, 1, 2], 'test')
acc2 = evaluate_class_network(network2, [8, 9], 'test', reverse=True)
model.logger.log('test_acc1', -1, acc1)
model.logger.log('test_acc2', -1, acc2)
pickle.dump(model.logger, open(f"examples/BURGLARY/results/burglary_e{epochs}_{seed}", "wb"))
