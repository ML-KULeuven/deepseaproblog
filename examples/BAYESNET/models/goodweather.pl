% Variables
pleasant(0, 0, S) ~ beta([9, 2]).
pleasant(0, 1, S) ~ beta([1, 1]).
pleasant(1, 0, S) ~ beta([11, 7]).
pleasant(1, 1, S) ~ beta([1, 9]).

temp(X, S) ~ normal(temp_net([X])).
cloudy(X, Y) ~ categorical(cloudy_net([X]), [0, 1, 2]).
humid(X, Y) ~ categorical(humid_net([X]), [8, 9]).

% Program
rainy(I1, I2) :-
    cloudy(I1, N), \+tf_eq(0, N), humid(I2, M), tf_eq(9, M).

good_weather(I1, I2, T) :-
    rainy(I1, I2), smaller_than(T, 0), pleasant(1, 0, S2), smaller_than(0.5, S2).
good_weather(I1, I2, T) :-
    rainy(I1, I2), \+smaller_than(T, 0), pleasant(1, 1, S2), smaller_than(0.5, S2).
good_weather(I1, I2, T) :-
    \+rainy(I1, I2), smaller_than(15, T), pleasant(0, 0, S2), smaller_than(0.5, S2).
good_weather(I1, I2, T) :-
    \+rainy(I1, I2), \+smaller_than(15, T), pleasant(0, 1, S2), smaller_than(0.5, S2).

P :: depressed(I1) :-
    member(N, [1, 2]),
    cloudy(I1, C), tf_eq(N, C), P is N * 0.2.

good_day(I1, I2, X) :-
    \+depressed(I1), temp(X, T),  good_weather(I1, I2, T).

direct_supervision(I1, I2, X, T, C, H) :-
    temp(X, S), equals(S, T), cloudy(I1, C1), tf_eq(C, C1), humid(I2, H1), tf_eq(H, H1).

temperature_supervision(I1, I2, X, T, C, H) :-
    temp(X, S), equals2(S, T).

