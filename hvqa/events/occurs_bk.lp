holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).

object(Id, Frame) :- holds(class(_, Id), Frame).

disappear(Id, -I) :-
  holds(class(Class, Id), I),
  not holds(class(Class, Id), -I),
  step(-I), step(I).
