from typing import TYPE_CHECKING, Sequence
from query import Query
from tensor import TensorStore, ModuleStore
from problog.formula import LogicFormula
from problog.logic import Term
from problog.program import LogicProgram

if TYPE_CHECKING:
    from model import Model


class Engine(object):
    """
    An asbtract engine base class.
    """

    def __init__(self, model: "Model"):
        """

        :param model: The model that this engine will solve queries for.
        """
        self.model = model
        self.tensor_store = TensorStore()
        self.module_store = ModuleStore()
        self.extra_parameters = {}

    def perform_count(self, queries: Sequence[Query], acs):
        pass

    def prepare(self, program: LogicProgram) -> LogicProgram:
        """
        Modifies the given program to a format suited for querying in this engine.
        :param program: The program to be modified
        :return: The modified program
        """
        raise NotImplementedError("prepare is an abstract method")

    def ground(self, query: Query, **kwargs) -> LogicFormula:
        """

        :param query: The query to ground.
        :param kwargs:
        :return: A logic formula representing the ground program.
        """
        raise NotImplementedError("ground is an abstract method")

    def register_foreign(self, func, function_name, arity):
        """
        Makes a Python function available to the grounding engine.
        :param func: The Python function to be made available.
        :param function_name: The name of the predicate that will be used to address this function in logic.
        :param arity: The arity of the function.
        :return:
        """
        raise NotImplementedError("register_foreign is an abstract method")

    def get_tensor(self, tensor_term: Term):
        if tensor_term.functor == "tensor":
            try:
                return self.tensor_store[tensor_term.args[0]]
            except KeyError as e:
                return tensor_term.args[0].value
        return tensor_term

    def get_hyperparameters(self) -> dict:
        raise NotImplementedError("get_hyperparameters is an abstract method")

    def train(self):
        """
        Set th engine to train mode.
        :return:
        """
        pass

    def get_extra_parameters(self):
        return list(self.extra_parameters.values())
