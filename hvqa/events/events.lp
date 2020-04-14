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
  not nn_error(F, I),
  not obs(F, false, I).

obs(Swap, true, I) :- swapped(Swap, Old, I).
obs(Old, false, I) :- swapped(Swap, Old, I).

% Error if a new object appears (which wasn't in initial frame)
error_obj(Id, Frame) :-
  obs(class(Class, Id), Frame),
  not obs(class(Class, Id), 0).

% If there are two octopuses there may be a tracker error
{ tracker_error(ErrorId, Id) } :-
  error_obj(ErrorId, Frame),
  obs(class(octopus, Id), Frame),
  ErrorId != Id.

% Create new fluents with correct Id but erroneous properties
swapped(colour(C1, Id), colour(C2, Id), I) :-
  tracker_error(ErrorId, Id),
  obs(colour(C1, ErrorId), I),
  obs(colour(C2, Id), I).

swapped(rotation(R1, Id), rotation(R2, Id), I) :-
  tracker_error(ErrorId, Id),
  obs(rotation(R1, ErrorId), I),
  obs(rotation(R2, Id), I).

swapped(position(P1, Id), position(P2, Id), I) :-
  tracker_error(ErrorId, Id),
  obs(position(P1, ErrorId), I),
  obs(position(P2, Id), I).

% If an object is detected in error, assume all properties are erroneous
nn_error(class(Class, Id), Frame) :-  error_obj(Id, Frame), obs(class(Class, Id), Frame).
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
