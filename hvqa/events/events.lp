% ***** File for finding which events occurred by modelling them as possible explanations *****

% Feature engineering
obs_exists(Id, Frame) :- obs(class(_, Id), true, Frame).

obs_disappear(Id, Frame) :-
  obs_exists(Id, Frame),
  occurs(move(_), Frame),
  not obs_exists(Id, Frame+1),
  step(Frame+1).

% Map to internal representation
obs(F, true, I) :- obs(F, I).

% Assume correct initial state
holds(F, 0) :- obs(F, true, 0).

% Defined fluents are hardcoded, so we need to copy them over to holds
holds(F, I) :-
  fluent(defined, F),
  obs(F, true, I).

% We only need CWA for exists
obs(exists(Id), false, I) :-
  fluent(inertial, exists(Id)),
  not obs(exists(Id), true, I),
  step(I).

% Reality check axioms
:- obs(class(octopus, Id), true, I), occurs(nothing(Id), I).
:- obs(class(octopus, Id), false, I), -occurs(nothing(Id), I).

:~ obs(exists(Id), false, I), holds(exists(Id), I). [1@1, exists(Id), I]
:~ obs(F, true, I), -holds(F, I). [1@2, F, I]

% Generate possible events
occurs(A, I) : action(A) :- step(I+1), I >= 0.

#show occurs/2.
#show try_obj/2.
