# DeepSeaProbLog

This is the official repository of [DeepSeaProbLog](https://proceedings.mlr.press/v216/de-smet23a.html).
DeepSeaProbLog is an extension of [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) that integrates logic-based reasoning with (deep)
probabilistic programming in discrete-continuous domains.

## Installation

## Tutorial

The unifying concept in DeepSeaProbLog is the *neural distributional fact*

  ```prolog
  variable([Inputs], S) ~ distribution(network([Inputs])).
  ```
  
Where `[Inputs]` is a list of (neural) inputs, `network` is the identifier of a neural network parametrising `distribution` and `S` represents `variable`. 
Note that parameters do not need to originate from a neural network and can also be directly put into the distribution as

  ```prolog
  variable(S) ~ distribution(Parameters).
  ```

For example, a two-dimensional standard normal distribution is defined as

  ```prolog
  normal_var(S) ~ normal([[0, 0], [1, 1]]).
  ```
  
A regular DeepSeaProbLog program then consists of a list of variables followed by a list of logical rules, for example

  ```prolog
  % Variables
  var_1(_, S) ~ distr_1(_).
  ...
  var_n(_, S) ~ distr_n(_).
  
  % Logic
  rule_1(_) :-
      var_1(_, V1), var_n(_, Vn), add(V1, Vn, Vsum), smaller_than(0, Vsum).
  ...
  rule_m(_) :- 
      var_1(_, V1), var_n(_, Vn), equals(V1, Vn).
  ```
  
A program should be stored as a `.pl` file and can be loaded into Python through the 
[`Model`](https://github.com/LennertDeSmet/DeepSeaProbLog/blob/dev/model.py) class. Apart from a `.pl` file,
the Model interface also takes a list of all neural networks used in the program. 
In total, we load a joint neural-symbolic, probabilistic `model` in Python as

  ```python
  from model import Model
  
  model = Model("program.pl", networks)
  ```

The probability of a query `q` can be computed directly via `model.solve_query(q)
`. 

For training purposes, the query has to be compiled, after which training can occur as traditional in a ML context

```python
model.set_optimizer(Adam(learning_rate=0.001))
model.set_loss(BinaryCrossentropy())
model.compile_query(q)
model.train(data)
```



## Experiments

All experiments from [our paper](https://proceedings.mlr.press/v216/de-smet23a.html) are provided in [DeepSeaProbLog/examples](https://github.com/LennertDeSmet/DeepSeaProbLog/tree/dev/examples), together
with a couple of novel tasks.

## 
