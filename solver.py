from time import time
from typing import Type, TYPE_CHECKING, Optional, List, Sequence
from arithmetic_circuit import ArithmeticCircuit
from engines import Engine
from query import Query
from semiring import Result
from semiring.tensor_semiring import TensorSemiring, Semiring
from problog.formula import LogicFormula
from problog.logic import Term

if TYPE_CHECKING:
    from model import Model


class Solver(object):
    """
    A class that bundles the different steps of inference.
    """

    def __init__(
        self,
        model: "Model",
        engine: Engine,
        semiring: Type[Semiring] = TensorSemiring
    ):
        """

        :param model: The model in which queries will be evaluated.
        :param engine: The engine that will be used to ground queries.
        :param semiring: The semiring that will be used to evaluate the arithmetic circuits.
        :param cache: If true, then arithmetic circuits will be cached.
        :param cache_root: If cache_root is not None, then the cache is persistent and is saved to that directory.
        """
        self.engine = engine
        self.model = model
        self.program = self.engine.prepare(model.program)
        self.semiring = semiring
        self.current_query = None

    def build_ac(self, q: Query) -> ArithmeticCircuit:
        """
        Builds the arithmetic circuit.
        :param q: The query for which to build the arithmetic circuit.
        :return: The arithmetic circuit for the given query.
        """
        self.current_query = q
        start = time()
        substitute = False
        ground = self.engine.ground(q, substitute=substitute, label=LogicFormula.LABEL_QUERY)
        ground_time = time() - start
        ac = ArithmeticCircuit(ground, self.semiring, ground_time=ground_time)
        self.current_query = None
        return ac

    def solve(self, batch: Sequence[Query]) -> List[Result]:
        """
        Performs inference for a batch of queries.
        :param batch: A list of queries to perform inference on.
        :return: A list of results for the given queries.
        """
        self.engine.tensor_store.clear()
        # Build ACs
        acs: List[ArithmeticCircuit] = [self.build_ac(q) for q in batch]
        # Evaluate ACs. Evaluate networks if necessary
        result = [
            ac.evaluate(self.model, batch[i].substitution) for i, ac in enumerate(acs)
        ]
        semirings = [r.semiring for r in result]
        self.engine.perform_count(batch, (acs, semirings))
        return result

    def get_tensor(self, term: Term):
        return self.engine.get_tensor(term)

    def get_hyperparameters(self) -> dict:
        parameters = dict()
        parameters["engine"] = self.engine.get_hyperparameters()
        parameters["semiring"] = self.semiring.__name__
        return parameters
