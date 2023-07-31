import time

from collections import defaultdict
from os import PathLike
from typing import Collection, Optional, Tuple, List, Mapping, Any, Sequence, Union
from keras.optimizers import Optimizer
from keras.losses import Loss
from engines.exact_engine import ExactEngine
from engines.tensor_ops import *
from engines.builtins import register_tensor_predicates
from problog.logic import term2list, Term, Var, Constant, InstantiationError, Or, list2term
from problog.program import PrologString, PrologFile, SimpleProgram, LogicProgram
from embeddings import TermEmbedder
from engines import Engine
from network import Network
from query import Query
from semiring import Result
from semiring.tensor_semiring import TensorSemiring
from solver import Solver
from utils import check_path
from utils.logger import Logger


class Model(object):
    def __init__(
        self,
        program_string: str,
        networks: List[Network] = None,
        pcf_functions: List[Network] = None,
        embeddings: Optional[TermEmbedder] = None,
        load: bool = True,
        gumbel_temperature: float = 2.,
        gumbel_alpha: float = 1e-4,
        nb_samples: int = 250
    ):
        """
        :param program_string: A string representing a DeepProbLog program or the path to a file containing a program.
        :param networks: A collection of networks that will be used to evaluate the neural predicates.
        :param embeddings: A TermEmbedder used to embed Terms in the program.
        :param load: If true, then it will attempt to load the program from 'program_string',
         else, it will consider program_string to be the program itself.
        """
        super(Model, self).__init__()
        if load:
            self.program: LogicProgram = PrologFile(program_string)
        else:
            self.program: LogicProgram = PrologString(program_string)

        if networks is None:
            networks = []
        self.networks = dict()
        for network in networks:
            self.add_network(network, det=True)

        if pcf_functions is None:
            pcf_functions = []
        self.pcf_functions = dict()
        for pcf_function in pcf_functions:
            self.add_pcf(pcf_function)

        self.nb_samples = nb_samples
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_alpha = gumbel_alpha
        self.add_builtins()

        self.embeddings = embeddings
        self.tensor_sources = dict()
        self.tensor_sources["inputs"] = dict()
        self.tensor_sources["mem_inputs"] = dict()
        self.optimizer = None
        self.loss = None
        self.graph = None
        self.memgraph = None
        self.logger = Logger()
        self.counter = 0
        self.set_engine(ExactEngine(self))

        self.breaker_input = tf.keras.Input(shape=())
        self.add_tensor_source("breaker", {0: self.breaker_input})

    def get_embedding(self, term: Term):
        return self.embeddings.get_embedding(term)

    def get_trainable_variables(self):
        return self.graph.trainable_variables

    def set_engine(self, engine: Engine, **kwargs):
        """
        Initializes the solver of this model with the given engine and additional arguments.
        :param engine: The engine that will be used to ground queries in this model.
        :param kwargs: Additional arguments passed to the solver.
        :return:
        """
        self.solver = Solver(self, engine, **kwargs)
        register_tensor_predicates(engine)


    def set_optimizer(self, optimiser: Optimizer):
        self.optimizer = optimiser

    def set_loss(self, loss: Loss):
        self.loss = loss

    def reset_logger(self):
        self.logger = Logger()

    def add_network(self, net, det=False):
        self.networks[net.name] = net
        net.model = self
        net.det = det

    def add_pcf(self, pcf):
        self.pcf_functions[pcf.name] = pcf
        pcf.model = self

    def add_builtins(self):
        self.add_network(Network(Is(), "tf_eq"), det=True)
        self.add_network(Network(ContinuousOperation(operator="add"), "add"), det=True)
        self.add_network(Network(ContinuousOperation(operator="sub"), "subtract"), det=True)
        self.add_network(Network(ContinuousOperation(operator="mul"), "mul"), det=True)
        self.add_network(Network(ContinuousOperation(operator="div"), "div"), det=True)
        self.add_network(Network(ContinuousOperation(operator="rounddiv"), "rounddiv"), det=True)
        self.add_network(Network(Sampler(self, sample_nb=self.nb_samples, temperature=self.gumbel_temperature,
                                         alpha=self.gumbel_alpha), "sampler_internal"), det=True)

        if "equals" not in self.networks.keys():
            self.add_network(Network(Equals(), "equals"), det=True)
        if "smaller_than" not in self.networks.keys():
            self.add_network(Network(SmallerThan(), "smaller_than"), det=True)
        if "unification" not in self.networks.keys():
            self.add_network(Network(SoftUnification(), "unification"), det=True)

    def evaluate_nn(self, to_evaluate: List[Tuple[Term, Term]]):
        """
        :param to_evaluate: List of neural predicates to evaluate
        :return: A dictionary with the elements of to_evaluate as keys, and the output of the NN as values.
        """
        result = dict()
        evaluations = defaultdict(list)
        # Group inputs per net to send in batch
        for net_name, inputs in to_evaluate:
            net = self.networks[str(net_name)]
            if net.det:
                tensor_name = Term("nn", net_name, inputs)
                if tensor_name not in self.solver.engine.tensor_store:
                    evaluations[net_name].append(inputs)
            else:
                if inputs in net.cache:
                    result[(net_name, inputs)] = net.cache[inputs]
                    del net.cache[inputs]
                else:
                    evaluations[net_name].append(inputs)
        for net in evaluations:
            network = self.networks[str(net)]
            out = network([term2list(x, False) for x in evaluations[net]])
            for i, k in enumerate(evaluations[net]):
                if network.det:
                    tensor_name = Term("nn", net, k)
                    self.solver.engine.tensor_store.store(out[i], tensor_name)
                else:
                    result[(net, k)] = out[i]
        return result

    def _solve(self,  query: Query, generate=False) -> List[Result]:
        if generate:
            self.breaker_input = tf.constant(1.)
            self.add_tensor_source("breaker", {0: self.breaker_input})
            generation = self.solver.solve([query])
            self.breaker_input = tf.keras.Input(shape=())
            self.add_tensor_source("breaker", {0: self.breaker_input})
            return generation
        else:
            return self.solver.solve([query])

    def _compile(self, query: Query, mem=False) -> None:
        """
            Builds the computation graph for the given query.
        :param query: The query to be solved and built.
        :param input: A list of Keras tensors that will be used as input to the computation graph.
        :param mem: If true, the graph of the query will be stored separately to be used as a memory query for
        continual learning or curriculum learning purposes.
        :return:
        """
        if mem:
            input = [v for k, v in self.tensor_sources["mem_inputs"].items()]
        else:
            input = [v for k, v in self.tensor_sources["inputs"].items()]
        input.append(self.breaker_input)
        start_time = time.time()
        outputs = list(self._solve(query)[0].result.values())
        graph = tf.keras.Model(inputs=input, outputs=outputs[0])
        print(f"Build time: {np.round(time.time() - start_time, 4)}s")
        if mem:
            self.memgraph = graph
        else:
            self.graph = graph
            self.counter = 0

    def _build_query(self, name: str, inputs, substitution, mem=False) -> Query:
        translated_input = []
        tensor_idx = {}
        for id, i in enumerate(inputs):
            if type(i) == Constant or type(i) == Var or type(i) == Term:
                translated_input.append(i)
            elif tf.keras.backend.is_keras_tensor(i) or tf.is_tensor(i):
                if mem:
                    translated_input.append(Term("tensor", Term("mem_inputs", Constant(id))))
                else:
                    translated_input.append(Term("tensor", Term("inputs", Constant(id))))
                tensor_idx[id] = i
            else:
                raise Exception(f"Invalid type of input {i}")
        if mem:
            self.add_tensor_source("mem_inputs", tensor_idx)
        else:
            self.add_tensor_source("inputs", tensor_idx)
        return Query(Term(name, *translated_input), substitution=substitution)

    def solve_query(self, query: str, inputs=None, substitution=None, generate=False) -> List[Result]:
        if inputs is None:
            inputs = []
        query = self._build_query(query, inputs, substitution)
        return self._solve(query, generate=generate)

    def compile_query(self, query: str, inputs=None, substitution=None, mem=False):
        if inputs is None:
            inputs = []
        query = self._build_query(query, inputs, substitution, mem)
        self._compile(query, mem=mem)

    def call(self, input, training=False, mem=False):
        input.append(tf.constant(1.))
        if mem:
            graph = self.memgraph
        else:
            graph = self.graph
        wmc = graph.call(input)
        if wmc.shape[-1] > 1:
            wmc = tf.reduce_mean(wmc, axis=-1)
        return wmc

    def save_state(self, filename: Union[str, PathLike]):
        """
            Save the compiled Tensorflow graph of a query to the given filename.
        :param filename: The filename to save the model to.
        :param complete: If true, save neural networks with information needed to resume training.
        :return:
        """
        check_path(filename)
        self.graph.save_weights(filename)


    def load_state(self, filename: Union[str, PathLike]):
        """
            Load the weights of a Tensorflow graph of a query from the given filename.
        :param filename: The filename to restore the model from.
        :return:
        """
        check_path(filename)
        self.graph.load_weights(filename)

    def grad(self, inputs, targets, training=False, mem_data=None):
        with tf.GradientTape() as tape:
            y_pred = self.call(inputs, training)
            probability = tf.squeeze(y_pred)
            loss_value = self.loss(targets, probability)

            # Include memory in a simple, continual learning replay fashion
            if mem_data is not None:    
                rd_id = np.random.randint(0, len(mem_data))
                mem_inputs, mem_targets = mem_data[rd_id][:-1], mem_data[rd_id][-1]
                mem_y_pred = self.call(mem_inputs, training, mem=True)
                loss_value += 5 * self.loss(mem_targets, mem_y_pred)
        return loss_value, tape.gradient(loss_value, self.graph.trainable_variables)

    def train(self, data, epochs, log_its=100, mem_data=None, val_data=None, eval_fns=None, fn_args=None, training=False):
        """
        Trains all weights present in the model graph.
        """
        if mem_data is not None:
            param_list = self.graph.trainable_variables
            param_list.extend(self.memgraph.trainable_variables)
        else:
            param_list = self.graph.trainable_variables

        for epoch in range(epochs):
            print("Epoch {}".format(epoch + 1))
            accumulated_loss = 0
            acc_eval_time = 0
            prev_iter_time = time.time()
            for el in data:
                x, y = el[:-1], el[-1]
                prev_eval_time = time.time()
                loss_val, grads = self.grad(x, y, training=training, mem_data=mem_data)
                self.optimizer.apply_gradients(zip(grads, param_list))
                acc_eval_time += time.time() - prev_eval_time
                accumulated_loss += loss_val.numpy()

                self.counter += 1
                if self.counter % log_its == 0:
                    update_time = time.time() - prev_iter_time
                    if val_data == None:
                        print(
                            "Iteration: ",
                            self.counter,
                            "\ts:%.4f" % update_time,
                            "\tAverage Loss: ",
                            accumulated_loss / log_its
                        )
                        self.log(self.counter, accumulated_loss, acc_eval_time, update_time, log_iter=log_its)
                    else:
                        val_loss = 0
                        val_counter = 0
                        for val_el in val_data:
                            x2, y2 = val_el[:-1], val_el[-1]
                            val_y = self.call(x2, training)
                            val_loss += self.loss(y2, val_y).numpy()
                            val_counter += 1
                        evals = []
                        if eval_fns is not None:
                            for id, fn in enumerate(eval_fns):
                                eval = fn(*fn_args[id])
                                self.logger.log(f"val_accs{id}", self.counter, eval)
                                evals.append(eval)
                        print(
                            "Iteration: ",
                            self.counter,
                            "\ts:%.4f" % update_time,
                            "\tAverage Loss: ",
                            accumulated_loss / log_its,
                            "\tValidation Loss: ",
                            val_loss / val_counter,
                            "\tValidation Eval: ",
                            evals
                        )
                        self.log(self.counter, accumulated_loss, acc_eval_time, update_time, log_iter=log_its)
                    accumulated_loss = 0
                    prev_iter_time = time.time()


    def register_foreign(self, *args, **kwargs):
        self.solver.engine.register_foreign(self.solver.program, *args, **kwargs)

    def __str__(self):
        return "\n".join(str(line) for line in self.program)

    def get_tensor(self, term: Term):
        """
        :param term: A term of the form tensor(_).
        If the tensor is of the form tensor(a(*args)), then it will look into tensor source a.
        :return:  Returns the stored tensor identifier by the term.
        """
        tensor_list = []
        if type(term) == list:
            for i in term:
                tensor_list.append(self.get_tensor_helper(i))
            return tensor_list
        else:
            return self.get_tensor_helper(term)

    def get_tensor_helper(self, term: Term) -> tf.Tensor:
        if type(term) == int:
            return term
        if len(term.args) > 0 and term.args[0].functor in self.tensor_sources:
            if len(term.args) > 0 and term.args[0].functor in self.tensor_sources:
                if type(term.args[0].args) is tuple:
                    return self.tensor_sources[term.args[0].functor][term.args[0].args[0].value]
                else:
                    return self.tensor_sources[term.args[0].functor][term.args[0].args]
        return self.solver.get_tensor(term)

    def store_tensor(self, tensor: tf.Tensor) -> Term:
        """
        Stores a tensor in the tensor store and returns and identifier.
        :param tensor: The tensor to store.
        :return: The Term that is the identifier by which this tensor can be uniquely identified in the logic.
        """
        return Term("tensor", Constant(self.solver.engine.tensor_store.store(tensor)))

    def add_tensor_source(
        self, name: str, source: Mapping[Any, tf.Tensor]
    ):
        """
        Adds a named tensor source to the model.
        :param name: The name of the added tensor source.
        :param source: The tensor source to add
        :return:
        """
        self.tensor_sources[name] = source

    def get_parameters(self):
        return self.graph.variables

    def log(
        self, counter, acc_loss, eval_timing, it_timing, snapshot_iter=None,
            log_iter=100, verbose=1, **kwargs
    ):
        if (
            "snapshot_name" in kwargs
            and snapshot_iter is not None
            and counter % snapshot_iter == 0
        ):
            filename = "{}_iter_{}.mdl".format(kwargs["snapshot_name"], counter)
            print("Writing snapshot to " + filename)
            self.save_state(filename)
        if verbose and counter % log_iter == 0:
            self.logger.log("time", counter, it_timing)
            self.logger.log("loss", counter, acc_loss / log_iter)
            self.logger.log("eval_time", counter, eval_timing / log_iter)
