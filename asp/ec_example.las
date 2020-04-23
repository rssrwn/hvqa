
% Mode Declarations

#modeh(initiatedAt(meeting(var(p1), var(p2)), var(frame))).
#modeh(initiatedAt(meeting(var(p1), var(p2)), var(frame))).

#modeb(happensAt(close(var(p1), var(p2), var(threshold)), var(frame))).
#modeb(happensAt(active(var(p1), var(frame)))).
#modeb(happensAt(inactive(var(p1), var(frame)))).
#modeb(happensAt(walking(var(p1), var(frame)))).
#modeb(happensAt(running(var(p1), var(frame)))).
#modeb(happensAt(disappear(var(p1), var(frame)))).


fluent(meeting(X,Y)) :- person(X), person(Y).


holdsAt(F,Te) :-
  fluent(F),
  initiatedAt(F,Ts),
  next_time(Ts, Te),
  time(Ts).

holdsAt(F,Te) :-
  fluent(F),
  holdsAt(F,Ts),
  not terminatedAt(F,Ts),
  next_time(Ts, Te),
  time(Ts).


happensAt(close(Id1,Id2,Threshold),Time) :-
  dist(Id1,Id2,Time,Distance),
  dist(Threshold), Distance <= Threshold.


dist(24).
dist(25).
dist(27).
dist(34).
dist(40).

time(1).
next_time(1, 2).


#pos(p_5697@1, {}, {}, {
  person(id0).
  :- holdsAt(A, 2), A = meeting(_, _), not goal(holdsAt(A, 2)).
  :- not holdsAt(A, 2), goal(holdsAt(A,2)).
  happensAt(inactive(id0),1).
}).


#pos(p_5556@1, {}, {}, {
  person(id0).
  :- holdsAt(A, 2), A = meeting(_, _), not goal(holdsAt(A, 2)).
  :- not holdsAt(A, 2), goal(holdsAt(A,2)).
  happensAt(walking(id0),1).
}).


#pos(p_2245@1, {}, {}, {
  person(id0).
  :- holdsAt(A, 2), A = meeting(_, _), not goal(holdsAt(A, 2)).
  :- not holdsAt(A, 2), goal(holdsAt(A,2)).
  happensAt(walking(id0),1).
}).

#pos(p_816@1, {}, {}, {
  person(id0).
  person(id1).
  person(id2).
  goal(holdsAt(meeting(id0,id1),2)).
  goal(holdsAt(meeting(id1,id0),2)).
  :- holdsAt(A, 2), A = meeting(_, _), not goal(holdsAt(A, 2)).
  :- not holdsAt(A, 2), goal(holdsAt(A,2)).
  happensAt(inactive(id0),1).
  happensAt(inactive(id1),1).
  happensAt(walking(id2),1).
  holdsAt(meeting(id0,id1),1).
  holdsAt(meeting(id1,id0),1).
  dist(id0,id1,1,19).
  dist(id1,id0,1,19).
  dist(id0,id2,1,289).
  dist(id2,id0,1,289).
  dist(id1,id2,1,293).
  dist(id2,id1,1,293).
}).
