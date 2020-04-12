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

occurs_event(Action, Frame) :- occurs(Action, Frame).

occurs_event(change_colour(Id), Frame) :- changed(colour, _, _, Id, Frame).

occurs_event(eat_fish(Octo), Frame) :-
  holds(class(fish, Fish), Frame),
  not exists(Fish, Frame+1),
  holds(class(octopus, Octo), Frame),
  holds(class(octopus, Octo), Frame+1).

occurs_event(eat_bag(Octo), Frame) :-
  holds(class(bag, Bag), Frame),
  not exists(Bag, Frame+1),
  holds(class(octopus, Octo), Frame),
  holds(class(octopus, Octo), Frame+1).
