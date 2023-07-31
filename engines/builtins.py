from problog.logic import Term, Constant, is_list, term2list, list2term
from engines.tensor_ops import *


def embed(engine, term):
    embedding = engine.model.get_embedding(term)[0, :]
    return Term("tensor", Constant(engine.tensor_store.store(embedding)))


def to_tensor(model, a):
    if type(a) is Term:
        if is_list(a):
            a = term2list(a)
        else:
            return model.get_tensor(a)
    if type(a) is list:
        out = [to_tensor(model, x) for x in a]
        return [x for x in out if x is not None]
    else:
        return float(a)


def tensor_wrapper(engine, func, *args):
    model = engine.model
    inputs = [to_tensor(model, a) for a in args]
    out = func(*inputs)
    return model.store_tensor(out)


def partial_tensor_wrapper(engine, func, *args):
    model = engine.model
    inputs = [to_tensor(model, a) for a in args]
    return func(*inputs)


def tf_mul(x, y):
    mul_op = Operator('mul')
    return mul_op.call(x, y)


def tf_add(x, y):
    add_op = Operator('add')
    return add_op.call(x, y)


def tf_subtract(x, y):
    sub_op = Operator('sub')
    return sub_op.call(x, y)

def tf_rounddiv(x, y):
    div_op = Operator('rounddiv')
    return div_op.call(x, y)


def tf_equals(target, options):
    """
    @:param target: the (batch_size, ) - tensor of target values.
    @:param options: a list of options and (batch_size, options) - tensor of probabilites
    @:return selects the probability of the desired target tf.squeeze(outcome from the options,
    this is a (batch_size, ) - tensor of probabilities.
    """
    is_obj = Is()
    return is_obj.call(target, options)


def argmax(options):
    outcomes, outcome_probs = options
    target = tf.argmax(outcome_probs, axis=-1)
    return tf.one_hot(target, axis=-1, depth=len(outcomes))


def symbolic_argmax(options):
    outcomes = options[0]
    outcome_probs = options[1]

    target = tf.argmax(outcome_probs, axis=-1)
    return Constant(int(target[0]))
    # return Constant(outcomes[int(target[0])])


def div(x, y):
    return x / y


def rounded_div(x, y):
    return tf.stop_gradient(x // y - x / y) + x / y


def sqrt(x):
    return tf.sqrt(x)


def matmul(A, y):
    return tf.matmul(A, y)


def dot(x, y):
    return tf.tensordot(x, y)


def sigmoid(x):
    return tf.math.sigmoid(x)


def sample(samples):
    rd_id = np.random.randint(0, samples.shape[-1])
    squeeze_dims = []
    for i in range(1, 3):
        if samples.shape[i] == 1:
            squeeze_dims.append(i)
    return tf.squeeze(samples[..., rd_id], axis=squeeze_dims)


def sort(x):
    return tf.sort(x, axis=-1)


def distance(x, y):
    return DistanceOp().call(x, y)


def concat(tensors):
    tf.stack(tensors, 0)
    return tf.concat(tensors, axis=0)


def select_comp(x, n):
    mult = tf.zeros_like(x)
    mult[n] = 1
    return tf.multiply(x, mult)


def select_dim(x, n):
    return x[:, int(n)]


def max(x):
    x = tf.stack(x, 0)
    x, _ = tf.reduce_max(x, 0)
    return x


def print_tensor(x):
    tf.print(x)
    return tuple()


def mean(x):
    x = tf.stack(x, 0)
    x = tf.reduce_mean(x, 0)
    return x


def one_hot(x, n):
    return tf.one_hot(x, axis=-1, depth=n)


def stack(tensors):
    return tf.stack(tensors)


def embeddings(engine, term):
    encoding = to_tensor(engine.model, term)
    encoding_size = math.prod(encoding.shape)

    encoding = encoding.view(-1, encoding_size)

    e1 = encoding[:, : encoding_size // 2]
    e2 = encoding[:, encoding_size // 2 :]

    e1 = Term("tensor", Constant(engine.tensor_store.store(e1)))
    e2 = Term("tensor", Constant(engine.tensor_store.store(e2)))
    return list2term([e1, e2])


def register_tensor_predicates(engine):
    engine.register_foreign(lambda *x: embed(engine, *x), "embed", (1, 0))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, sqrt, *x), "sqrt", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, argmax, *x), "argmax", (1, 1))
    engine.register_foreign(lambda *x: partial_tensor_wrapper(engine, symbolic_argmax, *x), "symbolic_argmax", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, sort, *x), "tf_sort", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, concat, *x), "concat", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, sample, *x), "sample", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, distance, *x), "distance", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, tf_subtract, *x), "tf_subtract", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, tf_add, *x), "tf_add", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, tf_mul, *x), "tf_mul", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, tf_rounddiv, *x), "tf_rounddiv", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, dot, *x), "dot", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, matmul, *x), "matmul", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, max, *x), "max", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, mean, *x), "mean", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, stack, *x), "stack", (1, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, one_hot, *x), "one_hot", (2, 1))
    engine.register_foreign(lambda *x: tensor_wrapper(engine, select_dim, *x), "select_dim", (2, 1))
    engine.register_foreign(lambda *x: embeddings(engine, *x), "embeddings", (1, 1))
    engine.register_foreign(lambda x: print_tensor(to_tensor(engine.model, x)), "print_tensor", (1, 0))
