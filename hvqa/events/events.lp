% ***** File for finding which events occurred by modelling them as possible explanations *****

% Map to representation used in book
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
:- obs(F, true, I), -holds(F, I).
:- obs(exists(Id), false, I), holds(exists(Id), I).

% Generate possible events
occurs(A, I) : action(A) :- step(I+1), I >= 0.

% wrong_obs_true(F, I) :- obs(F, true, I), -holds(F, I).
% wrong_obs_false(F, I) :- obs(F, false, I), holds(F, I).
% #show wrong_obs_true/2.
% #show wrong_obs_false/2.

% correct_actions :- occurs(move(0), 0).
% :- not correct_actions.

#show occurs/2.
% #show holds/2.
% #show disappear/2.
% #show exists/2.
