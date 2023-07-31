import pickle

from examples.BAYESNET.architectures import TemperaturePredictor2
from examples.BURGLARY.architectures import MNIST_CNN, MNIST_Net
from examples.BAYESNET.data import create_dataloaders
from model import Model
from network import Network
from engines.tensor_ops import *
from examples.BAYESNET.evaluate import *


seed = 7
epochs = 10

conv = MNIST_CNN()
network = MNIST_Net(conv, N=3)
network2 = MNIST_Net(conv, N=2)
temperature_net = TemperaturePredictor2(seed, 1)

smallerthan = SmallerThan(maxmult=24., scheme="constant", alpha=2e-4)
equals = Equals(maxmult=8., scheme="tanh", alpha=1e-3)

nets = []
nets.append(Network(network, "cloudy_net"))
nets.append(Network(network2, "humid_net"))
nets.append(Network(temperature_net, "temp_net"))
nets.append(Network(smallerthan, "smaller_than"))
nets.append(Network(equals, "equals"))

model = Model("examples/BAYESNET/models/goodweather.pl", nets)

# Set model loss and optimiser
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimiser = tf.keras.optimizers.Adamax(learning_rate=5e-3)
model.set_loss(loss)
model.set_optimizer(optimiser)

D = create_dataloaders("train", [[0, 1, 2], [8, 9]], batch_size=10, prob=True, full=0)
D_val = create_dataloaders("val", [[0, 1, 2], [8, 9]], batch_size=50, prob=True, full=0)
D_test = create_dataloaders("test", [[0, 1, 2], [8, 9]], batch_size=50, prob=True, full=2)

good_day_input = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]), tf.keras.Input(25)]
model.compile_query("good_day", good_day_input)
model.train(D, epochs, val_data=D_val,
            eval_fns=[evaluate_class_network, evaluate_class_network,
                      evaluate_temperature, evaluate_noise],
            fn_args=[[network, [0, 1, 2], 'val'], [network2, [8, 9], 'val', True, 10],
                     [temperature_net, D_test], [temperature_net]],
            log_its=100)

test_accs1 = evaluate_class_network(network, [0, 1, 2], 'test')
test_accs2 = evaluate_class_network(network2, [8, 9], 'val', True, 10)
test_tempdiff = evaluate_temperature(temperature_net, D_test)
test_noise = evaluate_noise(temperature_net)

model.logger.log("test_acs1", -1, test_accs1)
model.logger.log("test_acs2", -1, test_accs2)
model.logger.log("test_temp", -1, test_tempdiff)
model.logger.log("test_noise", -1, test_noise)
print(test_accs1, test_accs2, test_tempdiff, test_noise)

pickle.dump(model.logger, open(f"examples/BAYESNET/results/goodweather_e{epochs}_{seed}", "wb"))
