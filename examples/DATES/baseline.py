import tensorflow as tf
import time
import pickle
import multiprocessing
import os

from examples.DATES.architectures import AlexBaseline
from examples.DATES.data import create_dataloader
from examples.DATES.evaluate import *
from utils.logger import Logger
from examples.DATES.data.load_data import load_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPOCHS = 20
DIGITS = 4
CUR = "_cur"
MSE_LAMDA = 0.
LOG_ITS = 100
seed = 7

""" loading of datasets """
D, D_val, D_test, D_regressval, D_regresstest, D_cur = load_data(DIGITS, CUR)

alex_baseline = AlexBaseline(DIGITS)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
mse_loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-3)

logger = Logger()

""" Training loop """
acc_loss = 0
start_time = time.time()
for e in range(EPOCHS):
    print(f"Epoch {e + 1}")
    counter = 1
    for data in D:
        x, y1, y2, y3, y4 = data[0]
        x_cur, _, c1, c2, c3, c4 = D_cur[np.random.randint(0, len(D_cur))][0]
        ys = [y1, y2, y3, y4]
        cs = [c1, c2, c3, c4]
        with tf.GradientTape() as tape:
            y_preds, _ = alex_baseline.call(x)
            y_cur, mus = alex_baseline.call(x_cur)
            loss_val = 0
            for id, y_pred in enumerate(y_preds):
                loss_val += tf.reduce_mean(loss.call(ys[id], y_pred))
            for id, mu in enumerate(mus):
                loss_val += MSE_LAMDA * mse_loss(mu, cs[id])

            grads = tape.gradient(loss_val, alex_baseline.trainable_variables)
        acc_loss += loss_val
        optimizer.apply_gradients(zip(grads, alex_baseline.trainable_variables))
        if counter % LOG_ITS == 0:
            update_time = time.time() - start_time
            loss_val = 0
            for data in D_val:
                x, y1, y2, y3, y4 = data[0]
                ys = [y1, y2, y3, y4]
                with tf.GradientTape() as tape:
                    y_preds, sigmas = alex_baseline.call(x)
                    for id, y_pred in enumerate(y_preds):
                        loss_val += tf.reduce_mean(loss.call(ys[id], y_pred))
                    grads = tape.gradient(loss_val, alex_baseline.trainable_variables)
            accs = []
            fn_args = [[alex_baseline.classifier, alex_baseline.regressor, DIGITS, D_val], [alex_baseline.regressor, D_regressval, DIGITS]]
            for id, fn in enumerate([simple_evaluate_OD_classification, simple_evaluate_regression_IoU]):
                acc = fn(*fn_args[id])
                logger.log(f"val_accs{id}", counter, acc)
                accs.append(acc)
            print(
                "Iteration: ",
                counter,
                "\ts: %.4f" % update_time,
                "\tAverage Loss: %.4f" % (float(acc_loss) / LOG_ITS),
                "\tAverage Validation Loss: %.4f" % (float(loss_val) / len(D_val)),
                "\tValidation Eval: ", accs
                )
            logger.log("time", counter, update_time)
            logger.log("train_loss", counter, (float(acc_loss) / LOG_ITS))
            logger.log("val_loss", counter, (float(loss_val) / len(D_val)))
            start_time = time.time()
            acc_loss = 0
        counter += 1

ious = simple_evaluate_regression_IoU(alex_baseline.regressor, D_regresstest, DIGITS)
accs = simple_evaluate_OD_classification(alex_baseline.classifier, alex_baseline.regressor, DIGITS, D_test)
print(ious)
print(accs)
logger.log('test_iou', -1, ious)
logger.log('test_acc', -1, accs)

pickle.dump(logger,
            open(f"examples/DATES/results/baseline_{EPOCHS}e_10b_Adamax_MSE{MSE_LAMDA}_seed{seed}_barabas", "wb"))
