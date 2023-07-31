% Variables
earthquake(Earthq, Y) ~ categorical(earthq_net([Earthq]), [0, 1, 2]).
burglary(Burgl, Y) ~ categorical(burgl_net([Burgl]), [8, 9]).
loc(X, S) ~ normal(params([X])).

% Program
hears(X) :-
    loc(X, S), distance(0, S, D), smaller_than(D, 10).

0.9 :: alarm(_, Burgl) :-
    burglary(Burgl, Y), tf_eq(9, Y).
P :: alarm(Earthq, _) :-
    earthquake(Earthq, Y), member(N, [0, 1, 2]),
    tf_eq(N, Y), P is 0.35 * N.

calls(Earthq, Burgl, X) :- alarm(Earthq, Burgl), hears(X).
