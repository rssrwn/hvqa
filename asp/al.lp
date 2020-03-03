% ***** Types *****

% Timesteps
#const n = 4.
step(3..n).

#const num_objects = 3.
id(0..(num_objects-1)).

% Type definitions
fluent(inertial, rotation(R, Id)) :- property(rotation, R, Id, _).
fluent(inertial, position(P, Id)) :- property(position, P, Id, _).
fluent(inertial, class(C, Id)) :- property(class, C, Id, _).
fluent(inertial, colour(C, Id)) :- property(colour, C, Id, _).

fluent(defined, close(Id1, Id2)) :- relation(close, Id1, Id2, _).

% Assume only octopus can do actions
action(move(Id)) :- object(Id, octopus, _, _, _, _).
action(rotate_left(Id)) :- object(Id, octopus, _, _, _, _).
action(rotate_right(Id)) :- object(Id, octopus, _, _, _, _).


% ***** Domain Independent Axioms *****

% CWA for defined fluents
-holds(F, I) :-
  fluent(defined, F),
  not holds(F, I),
  step(I).

% CWA for actions
-occurs(A, I) :-
  not occurs(A, I),
  action(A),
  step(I).

% Inertia axioms
holds(F, I+1) :-
  fluent(inertial, F),
  holds(F, I),
  not -holds(F, I+1),
  step(I+1).

-holds(F, I+1) :-
  fluent(inertial, F),
  -holds(F, I),
  not holds(F, I+1),
  step(I+1),
  step(I).


% ***** Domain Dependent *****

% Definition for move
holds(position((X1, Y1 - 15, X2, Y2 - 15), Id), I+1) :-
  occurs(move(Id), I),
  holds(position((X1, Y1, X2, Y2), Id), I),
  holds(rotation(0, Id), I),
  step(I+1).

holds(position((X1 + 15, Y1, X2 + 15, Y2), Id), I+1) :-
  occurs(move(Id), I),
  holds(position((X1, Y1, X2, Y2), Id), I),
  holds(rotation(1, Id), I),
  step(I+1).

holds(position((X1, Y1 + 15, X2, Y2 + 15), Id), I+1) :-
  occurs(move(Id), I),
  holds(position((X1, Y1, X2, Y2), Id), I),
  holds(rotation(2, Id), I),
  step(I+1).

holds(position((X1 - 15, Y1, Y2 - 15, Y2), Id), I+1) :-
  occurs(move(Id), I),
  holds(position((X1, Y1, X2, Y2), Id), I),
  holds(rotation(3, Id), I),
  step(I+1).

% Definition for rotate
holds(rotation(NewR, Id), I+1) :-
  occurs(rotate_left(Id), I),
  holds(rotation(R, Id), I),
  NewR = R - 1,
  NewR >= 0,
  step(I+1).

holds(rotation(3, Id), I+1) :-
  occurs(rotate_left(Id), I),
  holds(rotation(R, Id), I),
  NewR = R - 1,
  NewR < 0,
  step(I+1).

holds(rotation(NewR, Id), I+1) :-
  occurs(rotate_right(Id), I),
  holds(rotation(R, Id), I),
  NewR = (R + 1) \ 4,
  step(I+1).

% Definition for colour
holds(colour(Col, Id), I) :-
  holds(class(octopus, Id), I),
  holds(close(Id, IdRock), I),
  holds(class(rock, IdRock), I),
  holds(colour(Col, IdRock), I).

% Executibility conditions
-occurs(move(Id), I) ; -occurs(rotate_left(Id), I) ; -occurs(rotate_right(Id), I) :-
  holds(class(octopus, Id), I),
  step(I).
