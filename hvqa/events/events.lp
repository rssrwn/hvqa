% ***** File for finding which events occurred by modelling them as possible explanations *****

% Feature engineering
obs_exists(Id, Frame) :- obs(class(_, Id), true, Frame).

obs_disappear(Id, Frame) :-
  obs_exists(Id, Frame),
  occurs(move(_), Frame),
  not obs_exists(Id, Frame+1),
  step(Frame+1).

% Map to internal representation if we think it is not an error
obs(F, true, I) :-
  obs(F, I),
  not nn_error(F, I).

% Error if a new object appears (which wasn't in initial frame)
nn_error(class(Class, Id), Frame) :-
  obs(class(Class, Id), Frame),
  not obs(class(Class, Id), 0).

% If there are two octopuses one of them is an error
1 { nn_error(class(octopus, Id1), Frame); nn_error(class(octopus), Id2) } 1 :-
  obs(class(octopus, Id1), Frame),
  obs(class(octopus, Id2), Frame),
  Id1 != Id2.

% If an object is detected in error, assume all properties are erroneous
error_obj(Id, Frame) :- nn_error(class(_, Id), Frame).
nn_error(colour(Colour, Id), Frame) :- error_obj(Id, Frame), obs(colour(Colour, Id), Frame).
nn_error(rotation(Rot, Id), Frame) :- error_obj(Id, Frame), obs(rotation(Rot, Id), Frame).
nn_error(position(Pos, Id), Frame) :- error_obj(Id, Frame), obs(position(Pos, Id), Frame).

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
:~ obs(exists(Id), false, I), holds(exists(Id), I). [1@1, exists(Id), I]
:~ obs(F, true, I), -holds(F, I). [1@2, F, I]

% Generate possible events
occurs(A, I) : action(A) :- step(I+1), I >= 0.

#show occurs/2.
