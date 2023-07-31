% Neural predicates
nn(proposal_net, [X], RegionList) :: region(X, RegionList).

% Variables
digit(Im, Params, Y) ~ categorical(classifier([Im, Params]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
box([Mu, Sigma], S) ~ generalisednormal([Mu, Sigma, 8.0]).

year_direct(Im, Year1, Year2, Year3, Year4) :-
    region(Im, [Y1, Y2, Y3, Y4]), 
    box(Y1, B1), box(Y4, B4),
    x_diff(0.0, B1, B1bound), smaller_than(B1bound, 0),
    x_diff(1.0, B4, B4bound), smaller_than(0, B4bound),
    ordered_output([Y1, Y2, Y3, Y4]),
    digit(Im, Y1, Ny1), digit(Im, Y2, Ny2), digit(Im, Y3, Ny3), digit(Im, Y4, Ny4),
    first_eq(Year1, Ny1), tf_eq(Year2, Ny2), tf_eq(Year3, Ny3), tf_eq(Year4, Ny4).

ordered_output([]).
ordered_output([[Mu, Sigma]]).
ordered_output([[Mu, Sigma], H2 | T]) :-
    box([Mu, Sigma], B1), box(H2, B2), 
    x_diff(B1, B2, Bdiff), smaller_than(Bdiff, 0),
    ordered_output([H2 | T]).
