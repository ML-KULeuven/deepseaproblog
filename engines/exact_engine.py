import tensorflow as tf

from numpy.random import choice
from engines.engine import Engine
from network import Network
from problog.engine import DefaultEngine
from problog.extern import problog_export
from problog.logic import (
    Term,
    AnnotatedDisjunction,
    term2list,
    Clause,
    And,
    Or,
    Var,
    Constant,
    list2term,
)
from problog.program import SimpleProgram

# EXTERN = '{}_extern_'
EXTERN = "{}_extern_nocache_"


def wrap_tensor(x, store):
    if type(x) is list:
        return list2term([wrap_tensor(e, store) for e in x])
    else:
        return Term("tensor", Constant(store.store(x)))


def create_with_substitution(formula, second, translation, key):
    if key in translation:
        return translation[key]
    node = second.get_node(key)
    t = type(node).__name__
    if t == "conj":
        return formula.add_and(
            [
                create_with_substitution(formula, second, translation, c)
                for c in node.children
            ],
            name=node.name,
        )
    elif t == "disj":
        return formula.add_or(
            [
                create_with_substitution(formula, second, translation, c)
                for c in node.children
            ],
            name=node.name,
        )
    else:
        raise (Exception("Unknown node ", node))


def get_predicate(net):
    def predicate(inputs):
        domain = net.domain
        output = net([term2list(inputs, False)])[0]
        net.cache[inputs] = output
        # if net.eval_mode:
        #     _, result = torch.topk(output, net.k, 0)
        #     result = [domain[int(r)] for r in result]
        # else:
        result = choice(
            domain, min(net.k, len(domain)), False, output.detach().numpy()
        ).tolist()
        return result

    return predicate


def translate_output(engine, output, tensor_name):
    if type(output) is list:
        translated_list = []
        for id, i in enumerate(output):
            tensor_name = Term("stored" + str(id), tensor_name)
            translated_list.append(translate_output(engine, i, tensor_name))
        return list2term(translated_list)
    elif type(output) is int or type(output) is float:
        return Constant(output)
    else:
        return Term("tensor", engine.tensor_store.store(output, tensor_name))


def get_det_predicate(net: Network, engine: Engine):
    if net.domain is None:
        def det_predicate(arguments):
            output = net([term2list(arguments)])[0]
            tensor_name = Term("nn", Term(net.name), arguments)
            return translate_output(engine, output, tensor_name)
    else:
        def det_predicate(arguments):
            output = net([term2list(arguments)])[0]
            tensor_name = Term("nn", Term(net.name), arguments)
            return list2term([net.domain, Term("tensor", engine.tensor_store.store(output, tensor_name))])
    return det_predicate


class ExactEngine(Engine):
    def __init__(self, model):
        Engine.__init__(self, model)
        self.breaker_term = Term("breaker")
        self.engine = DefaultEngine()
        self.supported_distributions = {"normal", "gamma", "beta", "uniform", "dirichlet", "poisson", "student",
                                        "logistic", "exponential", "generalisednormal", "categorical", "bernoulli",
                                        "gumbelsoftmax", "vae_decoder"}

    def prepare(self, db):
        translated = SimpleProgram()
        for e in db:
            new_es = [e]
            if type(e) is Term or type(e) is Clause:
                p = e.probability
                if e.functor in self.supported_distributions:
                    varterm = p
                    distterm = e
                    nn_param = distterm.args[0]
                    if distterm.functor == 'categorical':
                        if len(varterm.args) <= 1:
                            raise SyntaxError(
                                " Please use regular ProbLog syntax for a non-neural annotated disjunction! "
                            )
                        x = nn_param.args[0]
                        y = varterm.args[-1]
                        add_nn = Term("nn", nn_param.functor, x, y, distterm.args[1])
                        e_nn = varterm
                        e_nn.probability = add_nn
                        new_es = self.create_nn_predicate_ad(e_nn)
                    elif distterm.functor == 'bernoulli':
                        if len(varterm.args) <= 1:
                            raise SyntaxError(
                                " Please use regular ProbLog syntax for a non-neural, discrete fact! "
                            )
                        x = nn_param.args[0]
                        y = varterm.args[-1]
                        add_nn = Term("nn", nn_param.functor, x, y)
                        e_nn = varterm
                        e_nn.probability = add_nn
                        new_es = self.create_nn_predicate_fact(e)
                    elif distterm.functor == 'vae_decoder':
                        new_es = self.create_vae_predicate(varterm, distterm)
                    elif distterm.functor == 'gumbelsoftmax':
                        if type(distterm) is Clause:
                            raise SyntaxError(f" Let's not talk about dynamic random variables for now. ")
                        elif nn_param.functor == '.' or type(nn_param) == Var:
                                new_es = self.create_drv_predicate(varterm, distterm)
                        else:
                            new_es = self.create_nn_drv_predicate(varterm, distterm)
                    else:
                        if type(distterm) is Clause:
                            raise SyntaxError(f" Let's not talk about dynamic random variables for now. ")
                        elif nn_param.functor == '.' or type(nn_param) == Var:
                            new_es = self.create_crv_predicate(varterm, distterm)
                        else:
                            new_es = self.create_nn_crv_predicate(varterm, distterm)
                elif p is not None and p.functor == "nn":
                    if len(p.args) == 4:
                        new_es = self.create_nn_predicate_ad(e)
                    elif len(p.args) == 3:
                        new_es = self.create_nn_predicate_det(e)
                    elif len(p.args) == 2:
                        new_es = self.create_nn_predicate_fact(e)
                    else:
                        raise ValueError(
                            "A neural predicate with {} arguments is not supported.".format(
                                len(p.args)
                            )
                        )
            for new_e in new_es:
                translated.add_clause(new_e)
        translated.add_clause(
            Clause(
                Term("_directive"),
                Term("use_module", Term("library", Term("lists.pl"))),
            )
        )

        # Addition of arithmetic built-ins and tensorisation
        self.add_builtin_predicate(translated, "equals")
        self.add_builtin_predicate(translated, "smaller_than")
        self.add_builtin_predicate(translated, "unification")
        self.add_builtin_predicate(translated, "add", out=True)
        self.add_builtin_predicate(translated, "subtract", out=True)
        self.add_builtin_predicate(translated, "mul", out=True)
        self.add_builtin_predicate(translated, "div", out=True)
        self.add_builtin_predicate(translated, "rounddiv", out=True)
        self.add_builtin_predicate(translated, "rounddiv_d", out=True)
        self.add_builtin_predicate(translated, "tf_eq")
        # self.add_builtin_predicate(translated, "tf_add", out=True)
        # self.add_builtin_predicate(translated, "tf_subtract", out=True)
        # self.add_builtin_predicate(translated, "tf_mul", out=True)
        # self.add_builtin_predicate(translated, "tf_smaller")

        for pcf in self.model.pcf_functions:
            pcf_in = [Var(f"pcf_in{i}") for i in range(self.model.pcf_functions[pcf].inputs)]
            pcf_out = Var("pcf_out")
            pcf_head = Term(pcf, *pcf_in, pcf_out)
            pcf_body = Term(EXTERN.format(pcf), list2term(pcf_in), pcf_out)
            translated.add_clause(Clause(pcf_head, pcf_body))

        clause_db = self.engine.prepare(translated)
        problog_export.database = clause_db
        for network in self.model.networks:
            if self.model.networks[network].det:
                signature = ["+term", "-term"]
                func = get_det_predicate(self.model.networks[network], self)
                problog_export(*signature)(
                    func, funcname=EXTERN.format(network), modname=None
                )
            elif self.model.networks[network].k is not None:
                signature = ["+term", "-list"]
                problog_export(*signature)(
                    get_predicate(self.model.networks[network]),
                    funcname="{}_extern_nocache_".format(network),
                    modname=None,
                )
        
        for pcf in self.model.pcf_functions:        
            signature = ["+term", "-term"]
            func = get_det_predicate(self.model.pcf_functions[pcf], self)
            problog_export(*signature)(
                func, funcname=EXTERN.format(pcf), modname=None
            )

        return clause_db

    def ground(self, query, label=None, repeat=1, substitute=False, **kwargs):
        db = self.model.solver.program
        if substitute:
            query = query.substitute()
        ground = self.engine.ground(db, query.query, label=label)
        return ground

    def add_builtin_predicate(self, translated, name, out=False):
        in1 = Var(name + "_in1")
        in2 = Var(name + "_in2")
        p = Var(name + "_prob")
        if out:
            head = Term(name, in1, in2, p)
        else:
            head = Term(name, in1, in2)
            head.probability = p
        body = Term(EXTERN.format(name), list2term([in1, in2]), p)
        translated.add_clause(Clause(head, body))

    def create_drv_predicate(self, varterm, distterm):
        logits = distterm.args[0]
        classes = distterm.args[1]
        params = list2term([logits.args[0], classes])
        y = varterm.args[-1]

        head = varterm
        body = Term(EXTERN.format('sampler_internal'), list2term([params, distterm.functor]), y)
        return [Clause(head, body)]

    def create_nn_drv_predicate(self, varterm, distterm):
        nn_param = distterm.args[0]
        x = nn_param.args[0]
        logits = Var(nn_param.functor + '_paramvar')
        classes = distterm.args[1]
        params = list2term([logits, classes])
        y = varterm.args[-1]

        # Set up the translation to parameter and sampler calling
        head = varterm
        body_params = Term(EXTERN.format(nn_param.functor), x, logits)
        body_sampler = Term(EXTERN.format('sampler_internal'), list2term([params, distterm.functor]), y)
        return [Clause(head, And(body_params, body_sampler))]

    def create_crv_predicate(self, varterm, distterm):
        params = distterm.args[0]
        y = varterm.args[-1]
        breaker_term = Term("tensor", Term("breaker", Constant(0)))
        dist_term = list2term([distterm.functor, breaker_term])

        head = varterm
        body = Term(EXTERN.format('sampler_internal'), list2term([params, dist_term]), y)
        return [Clause(head, body)]

    def create_nn_crv_predicate(self, varterm, distterm):
        nn_param = distterm.args[0]
        x = nn_param.args[0]
        params = Var(nn_param.functor + '_paramvar')
        y = varterm.args[-1]

        head = varterm
        body_params = Term(EXTERN.format(nn_param.functor), x, params)
        body_sampler = Term(EXTERN.format('sampler_internal'), list2term([params, distterm.functor]), y)
        return [Clause(head, And(body_params, body_sampler))]

    def create_vae_predicate(self, varterm, distterm):
        nn_decoder = distterm.args[0]
        decoder_input = nn_decoder.args[0]
        latent = decoder_input.args[0]
        sample = Var(nn_decoder.functor + '_sample')
        y = varterm.args[-1]

        # Set up the uniform sampling and calling of decoder
        head = varterm
        body_sample = Term("sample", latent, sample)
        decoder_in_conditioned = [sample] + term2list(decoder_input)[1:]
        body_decoder = Term(EXTERN.format(nn_decoder.functor), list2term(decoder_in_conditioned), y)
        return [Clause(head, And(body_sample, body_decoder))]


    def create_nn_predicate_fact(self, e):
        p = e.probability
        net, inputs = p.args
        network = self.model.networks[str(net)]
        return [e]

    def create_nn_predicate_det(self, e):
        p = e.probability
        net, inputs, output = p.args
        network = self.model.networks[str(net)]
        network.det = True
        if network.k is not None:
            raise ValueError(
                "k should be None for deterministic network {}".format(str(net))
            )
        head = e.with_probability(None)
        body = Term(EXTERN.format(net), inputs, output)
        return [Clause(head, body)]

    def create_nn_predicate_ad(self, e):
        p = e.probability
        net, inputs, output, domain = p.args
        network = self.model.networks[str(net)]
        network.det = True
        network.domain = domain
        head = e.with_probability(None)
        body = Term(EXTERN.format(net), inputs, output)
        return [Clause(head, body)]

    def register_foreign(self, func, function_name, arity):
        signature = ["+term"] * arity[0] + ["-term"] * arity[1]
        problog_export.database = self.model.solver.program
        problog_export(*signature)(func, funcname=function_name, modname=None)

    def get_hyperparameters(self) -> dict:
        return {"type": "ExactEngine"}
