import pickle
import multiprocessing
import os
import time

from examples.DATES.architectures import *
from model import Model
from network import Network, PCF
from query import Query
from engines.tensor_ops import *
from problog.logic import Term, Constant
from examples.DATES.evaluate import *
from examples.DATES.data.load_data import load_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPOCHS = 10
DIGITS = 4
CUR = "_cur"
seed = 7

backbone = AlexConvolutions()
rpn = AlexRegressor(backbone, blocks=DIGITS)
classifier = AlexGeneralisedClassifier(backbone, conv=True)
equalsop = Equals(minmult=1e-2, maxmult=4., scheme="constant", alpha=DIGITS * 5e-3)
smallerop = SmallerThan(minmult=50., maxmult=50., scheme="constant", alpha= DIGITS * 1e-5)

nets = []
nets.append(Network(rpn, "proposal_net"))
nets.append(Network(classifier, "classifier"))
nets.append(Network(equalsop, "equals"))
nets.append(Network(smallerop, "smaller_than"))

x_diff = HorizontalDifference()

pcfs = []
pcfs.append(PCF(x_diff, 2, "x_diff"))

model = Model("examples/DATES/models/dates.pl", nets, pcf_functions=pcfs, nb_samples=250)

# Set model loss and optimiser
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimiser = tf.keras.optimizers.Adamax(learning_rate=5e-4)
model.set_loss(loss)
model.set_optimizer(optimiser)

# Load data
D, D_val, D_test, D_regressval, D_regresstest, D_cur = load_data(DIGITS, CUR)

year_direct_input = [tf.keras.Input([120, 200, 1])] + 4 * [tf.keras.Input([1])]
model.compile_query("year_direct", year_direct_input)
model.train(D, EPOCHS, val_data=D_val,
            eval_fns=[simple_evaluate_OD_classification, simple_evaluate_regression_IoU],
            fn_args=[[classifier, rpn, DIGITS, D_val], [rpn, D_regressval, DIGITS]])

ious = simple_evaluate_regression_IoU(rpn, D_regresstest, DIGITS)
accs = simple_evaluate_OD_classification(classifier, rpn, DIGITS, D_test)
print(ious)
print(accs)
model.logger.log('test_iou', -1, ious)
model.logger.log('test_acc', -1, accs)

pickle.dump(model.logger, open(f"examples/DATES/results/dates_e{EPOCHS}_{seed}", "wb"))