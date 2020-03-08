holds(P, Frame) :- obs(P, Frame).

holds(close(0, 2), 4).

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

holds_start(P) :- holds(P, 3).
holds_next(P) :- holds(P, 4).

#show occurs/2.
#show holds/2.
#show -holds/2.
