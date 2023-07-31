import matplotlib.pyplot as plt

from examples.LOGICVAE.architectures import *
from examples.LOGICVAE.data import create_simple_dataloaders
from model import Model
from network import Network
from engines.tensor_ops import *
from problog.logic import Constant, Var


seed = 0
epochs = 2
SHAPE_LATENT_DIM = 4
LOGIC_LATENT_DIM = 4
SAMPLE_SIZE = 50

encoder = Encoder(latent_dim=SHAPE_LATENT_DIM)
classifier = DigitClassifier(latent_dim=LOGIC_LATENT_DIM)
decoder = Decoder(latent_dim=SHAPE_LATENT_DIM + 10)

nets = []
nets.append(Network(encoder, "encoder_net"))
nets.append(Network(decoder, "decoder_net"))
nets.append(Network(classifier, "mnist_class"))

model = Model("examples/LOGICVAE/models/operation_generation.pl", nets, nb_samples=SAMPLE_SIZE)

D = create_simple_dataloaders("train", batch_size=10)

classifier.load_weights("examples/LOGICVAE/networks/classifier")
encoder.load_weights("examples/LOGICVAE/networks/shape_encoder")
decoder.load_weights("examples/LOGICVAE/networks/decoder")

# Generation of 2 images given a difference value
R = model.solve_query("generate", [Constant(5), Var("X1"), Var("X2")], generate=True)
number = 0
queries = []
for k in R[0].result.keys():
    queries.append(k)
sample = tf.random.uniform([], 0, len(queries), dtype=tf.int64)
generation1 = model.get_tensor(queries[sample.numpy()].args[-2])
generation2 = model.get_tensor(queries[sample.numpy()].args[-1])

plt.imshow(generation1[number], cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(generation2[number], cmap='gray')
plt.axis('off')
plt.show()


# Conditional generation in same handwriting style
given_n = 333  # 222, 188, 333

# Set encoder to eval mode to exactly match styles of the digits
encoder.eval = True
R = model.solve_query("generate_conditional", [Constant(0), D[given_n][0][0], Var("X1")], generate=True)
number = 0
print(R)
queries = []
for k in R[0].result.keys():
    queries.append(k)
generation = model.get_tensor(queries[0].args[-1])

plt.imshow(D[given_n][0][0][number], cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(generation[number], cmap='gray')
plt.axis('off')
plt.show()
