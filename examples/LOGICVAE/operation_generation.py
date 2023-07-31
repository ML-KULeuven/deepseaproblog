from examples.LOGICVAE.architectures import *
from examples.LOGICVAE.data import create_simple_dataloaders
from model import Model
from network import Network
from engines.tensor_ops import *


seed = 0
epochs = 2
SHAPE_LATENT_DIM = 4
LOGIC_LATENT_DIM = 4
SAMPLE_SIZE = 50

classifier = DigitClassifier(latent_dim=LOGIC_LATENT_DIM)
encoder = DenseEncoder(latent_dim=SHAPE_LATENT_DIM)
decoder = DenseDecoder()

equals_op = Equals(maxmult=0.2)
unif_op = SoftUnification(maxmult=4., n=1)

nets = []
nets.append(Network(encoder, "encoder_net"))
nets.append(Network(decoder, "decoder_net"))
nets.append(Network(classifier, "mnist_class"))
nets.append(Network(equals_op, "equals"))
nets.append(Network(unif_op, "unification"))

model = Model("examples/LOGICVAE/models/operation_generation.pl", nets, nb_samples=SAMPLE_SIZE)

# Set loss function and optimiser
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
model.set_loss(loss)
model.set_optimizer(optimiser)

# Curriculum phase to align digits with human interpretation
D_cur = create_simple_dataloaders("train", curriculum=True, batch_size=4, data_size=256)

curriculum_input = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]), tf.keras.Input([1]), tf.keras.Input([1])]
model.compile_query("image_subtraction_curriculum", curriculum_input)
model.train(D_cur, 1, log_its=10)

# Joint digit generation and classification training
D = create_simple_dataloaders("train", batch_size=10)
D_val = create_simple_dataloaders("val", batch_size=50)

gen_input = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]), tf.keras.Input([1])]
model.compile_query("encode_decode_subtraction", gen_input)
optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
model.set_optimizer(optimiser)
model.train(D, 2, val_data=D_val)

classifier.save_weights("examples/LOGICVAE/networks/classifier")
encoder.save_weights("examples/LOGICVAE/networks/shape_encoder")
decoder.save_weights("examples/LOGICVAE/networks/decoder")
