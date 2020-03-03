holds(class(Class, Id), 3) :- property(class, Class, Id, 3).
holds(colour(Colour, Id), 3) :- property(colour, Colour, Id, 3).
holds(rotation(Rot, Id), 3) :- property(rotation, Rot, Id, 3).
holds(position(Pos, Id), 3) :- property(position, Pos, Id, 3).

holds(close(0, 2), I) :- holds(position((25, 60, 45, 70), 0), I).

goal(I) :-
  holds(class(octopus, 0), I),
  holds(colour(blue, 0), I),
  holds(rotation(1, 0), I),
  holds(position((25, 60, 45, 70), 0), I),
  holds(class(fish, 1), I),
  holds(colour(silver, 1), I),
  holds(rotation(3, 1), I),
  holds(position((60, 100, 65, 105), 1), I),
  holds(class(rock, 2), I),
  holds(colour(blue, 2), I),
  holds(rotation(0, 2), I),
  holds(position((30, 60, 35, 65), 2), I).

success :- goal(I), I <= n.
:- not success.

1{occurs(A, I) : action(A)}1 :-
  step(I),
  not goal(I),
  I < n.

#show occurs/2.
