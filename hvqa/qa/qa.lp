% Show only the answer to the question in the answer set

answered :- answer(P, B, A).
answered :- answer(P, V).
answered :- answer(V).
:- not answered.

#show answer/1.
#show answer/2.
#show answer/3.

#show answered/0.
