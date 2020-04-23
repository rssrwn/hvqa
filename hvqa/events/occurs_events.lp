
% Map to holds (in case we need an abstraction)
holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

octo_id(Id) :- obs(class(octopus, Id), _).


% ***** Definitions for move *****

occurs(move(Id), Frame) :-
  holds(rotation(0, Id), Frame),
  holds(position((X1_1, Y1_1, X2_1, Y2_1), Id), Frame),
  holds(position((X1_2, Y1_2, X2_2, Y2_2), Id), Frame+1),
  Y1_1 > Y1_2,
  Y2_1 > Y2_2.

occurs(move(Id), Frame) :-
  holds(rotation(1, Id), Frame),
  holds(position((X1_1, Y1_1, X2_1, Y2_1), Id), Frame),
  holds(position((X1_2, Y1_2, X2_2, Y2_2), Id), Frame+1),
  X1_1 < X1_2,
  X2_1 < X2_2.

occurs(move(Id), Frame) :-
  holds(rotation(2, Id), Frame),
  holds(position((X1_1, Y1_1, X2_1, Y2_1), Id), Frame),
  holds(position((X1_2, Y1_2, X2_2, Y2_2), Id), Frame+1),
  Y1_1 < Y1_2,
  Y2_1 < Y2_2.

occurs(move(Id), Frame) :-
  holds(rotation(3, Id), Frame),
  holds(position((X1_1, Y1_1, X2_1, Y2_1), Id), Frame),
  holds(position((X1_2, Y1_2, X2_2, Y2_2), Id), Frame+1),
  X1_1 > X1_2,
  X2_1 > X2_2.


% ***** Definitions for rotate left *****

occurs(rotate_left(Id), Frame) :-
  holds(rotation(0, Id), Frame),
  holds(rotation(3, Id), Frame+1).

occurs(rotate_left(Id), Frame) :-
  holds(rotation(1, Id), Frame),
  holds(rotation(0, Id), Frame+1).

occurs(rotate_left(Id), Frame) :-
  holds(rotation(2, Id), Frame),
  holds(rotation(1, Id), Frame+1).

occurs(rotate_left(Id), Frame) :-
  holds(rotation(3, Id), Frame),
  holds(rotation(2, Id), Frame+1).


% ***** Definitions for rotate right *****

occurs(rotate_right(Id), Frame) :-
  holds(rotation(0, Id), Frame),
  holds(rotation(1, Id), Frame+1).

occurs(rotate_right(Id), Frame) :-
  holds(rotation(1, Id), Frame),
  holds(rotation(2, Id), Frame+1).

occurs(rotate_right(Id), Frame) :-
  holds(rotation(2, Id), Frame),
  holds(rotation(3, Id), Frame+1).

occurs(rotate_right(Id), Frame) :-
  holds(rotation(3, Id), Frame),
  holds(rotation(0, Id), Frame+1).


% Definitions for nothing action

% occurs(nothing(Id), Frame) :-
%   not occurs(move(Id), Frame),
%   not occurs(rotate_left(Id), Frame),
%   not occurs(rotate_right(Id), Frame),
%   step(Frame+1),
%   step(Frame),
%   octo_id(Id).

#show occurs/2.
