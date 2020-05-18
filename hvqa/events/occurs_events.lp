
% Map to holds (in case we need an abstraction)
holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

octo_id(Id) :- obs(class(octopus, Id), _).

obj_pos((X, Y), Id, Frame) :- holds(position((X, Y, _, _), Id), Frame).

disappear(Id, Frame+1) :-
  holds(class(Class, Id), Frame),
  not holds(class(Class, Id), Frame+1),
  step(Frame+1),
  step(Frame).


% ***** Definitions for move *****

occurs(move(Id), Frame) :-
  holds(rotation(0, Id), Frame),
  obj_pos((X1, Y1), Id, Frame),
  obj_pos((X2, Y2), Id, Frame+1),
  Y1 > Y2.

occurs(move(Id), Frame) :-
  holds(rotation(1, Id), Frame),
  obj_pos((X1, Y1), Id, Frame),
  obj_pos((X2, Y2), Id, Frame+1),
  X2 > X1.

occurs(move(Id), Frame) :-
  holds(rotation(2, Id), Frame),
  obj_pos((X1, Y1), Id, Frame),
  obj_pos((X2, Y2), Id, Frame+1),
  Y2 > Y1.

occurs(move(Id), Frame) :-
  holds(rotation(3, Id), Frame),
  obj_pos((X1, Y1), Id, Frame),
  obj_pos((X2, Y2), Id, Frame+1),
  X1 > X2.

% If the octopus disappears it must have moved
occurs(move(Id), Frame) :-
  holds(class(octopus, Id), Frame),
  disappear(Id, Frame+1).


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

occurs(nothing(Id), Frame) :-
  not occurs(move(Id), Frame),
  not occurs(rotate_left(Id), Frame),
  not occurs(rotate_right(Id), Frame),
  step(Frame+1),
  step(Frame),
  octo_id(Id).

#show occurs/2.
