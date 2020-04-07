% Feature engineering for QA (not event detection)

holds(F, I) :- obs(F, I).

changed(colour, Before, After, Id, Frame) :-
  holds(colour(Before, Id), Frame),
  holds(colour(After, Id), Frame+1),
  Before != After.

changed(rotation, Before, After, Id, Frame) :-
  holds(rotation(Before, Id), Frame),
  holds(rotation(After, Id), Frame+1),
  Before != After.

exists(Id, Frame) :- holds(class(_, Id), Frame).
