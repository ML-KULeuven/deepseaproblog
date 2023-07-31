% Variables
shape(X, S) ~ normal(encoder_net([X])).
generation(Latent, Condition, Gen) ~ vae_decoder(decoder_net([Latent, Condition])).
digit(X, Y) ~ categorical(mnist_class([X]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
prior(ID, S) ~ normal([[0, 0, 0, 0], [1, 1, 1, 1]]).

% Program
%% Optimisation logic
image_subtraction_curriculum(Image1, Image2, N1, N2) :-
    digit(Image1, D1), digit(Image2, D2),
    tf_eq(N1, D1), tf_eq(N2, D2).

encode_decode_subtraction(Image1, Image2, Diff) :-
    digit(Image1, D1), digit(Image2, D2),
    tf_subtract(D1, D2, R), tf_eq(Diff, R),
    argmax(D1, P1), argmax(D2, P2),
    shape(Image1, Shape1), shape(Image2, Shape2),
    generation(Shape1, P1, Gen1), generation(Shape2, P2, Gen2),
    prior(_, Prior1), prior(_, Prior2),
    equals(Shape1, Prior1), equals(Shape2, Prior2),
    unification(Gen1, Image1), unification(Gen2, Image2).

%% Generation logic
generate(Diff, GenLeft, GenRight) :-
    member(D1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    prior(_, Prior1), prior(_, Prior2),
    Diff is D1 - D2, generation(Prior1, D1, GenLeft), generation(Prior2, D2, GenRight).

generate_conditional(Diff, ImLeft, GenRight) :- 
    digit(ImLeft, DLeft), symbolic_argmax(DLeft, D1),
    member(D2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), Diff is D1 - D2,
    shape(ImLeft, ShapeLeft), generation(ShapeLeft, D2, GenRight).
