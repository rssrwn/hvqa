#const frame = 3.
#const next_frame = 4.

% Q1
answer(V) :-
  holds(colour(V, Id), Frame),
  holds(class(octopus, Id), Frame).

% Q2
% related :-
%   relation(close, Id1, Id2, frame),
%   object(Id1, octopus, _, _, _, frame),
%   object(Id2, rock, _, _, _, frame).
%
% answer(yes) :- related.
% answer(no) :- not related.

% Q3
% answer(A) :-
%  occurred(A, frame, next_frame).

% Q4
% answer(Prop, Val) :-
%   changed(Prop, _, Val, Id, frame),
%   object(Id, octopus, _, _, _, frame),
%   exists(Id, next_frame).
%
% answer(exists, false) :-
%   object(Id, octopus, _, _, _, frame),
%   exists(Id, frame),
%   not exists(Id, next_frame).

answered :- answer(P, V).
answered :- answer(V).
:- not answered.
#show answer/1.
#show answer/2.
