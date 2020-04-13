% Feature engineering for QA (not event detection)

step(I) :- obs(_, I).

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

action(move).
action(rotate_left).
action(rotate_right).
action(nothing).

event(A) :- action(A).
event(change_colour).
event(eat_fish).
event(eat_bag).

occurs_event(move, Id, Frame) :- occurs(move(Id), Frame).
occurs_event(rotate_left, Id, Frame) :- occurs(rotate_left(Id), Frame).
occurs_event(rotate_right, Id, Frame) :- occurs(rotate_right(Id), Frame).
occurs_event(nothing, Id, Frame) :- occurs(nothing(Id), Frame).

occurs_event(change_colour, Id, Frame) :- changed(colour, _, _, Id, Frame).

occurs_event(eat_fish, Octo, Frame) :-
  holds(class(fish, Fish), Frame),
  not exists(Fish, Frame+1),
  holds(class(octopus, Octo), Frame),
  holds(class(octopus, Octo), Frame+1).

occurs_event(eat_bag, Octo, Frame) :-
  holds(class(bag, Bag), Frame),
  not exists(Bag, Frame+1),
  holds(class(octopus, Octo), Frame),
  not exists(Octo, Frame+1),
  step(Frame+1).

event_count(Event, Id, Num) :-
  Num = #count { Event, Id, Frame : occurs_event(Event, Id, Frame) },
  occurs_event(_, Id, _),
  event(Event).
