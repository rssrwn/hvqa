% Show only the answer to the question in the answer set

answered :- answer(Idx, Prop, Before, After).
answered :- answer(Idx, Prop, Value).
answered :- answer(Idx, Ans).
:- not answered.

#show answer/2.
#show answer/3.
#show answer/4.
