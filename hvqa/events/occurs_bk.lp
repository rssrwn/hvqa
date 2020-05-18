holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).

object(Id, Frame) :- holds(class(_, Id), Frame).

disappear(Id, I+1) :-
  holds(class(Class, Id), I),
  not holds(class(Class, Id), I+1),
  step(I+1), step(I).

non_static_obj(Id) :- static(Id, Frame, false), step(Frame).

occurs(nothing(Id), Frame) :-
  not occurs(move(Id), Frame),
  not occurs(rotate_left(Id), Frame),
  not occurs(rotate_right(Id), Frame),
  non_static_obj(Id),
  step(Frame+1),
  step(Frame).
