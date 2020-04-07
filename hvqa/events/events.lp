% ***** File for finding which events occurred by modelling them as possible explanations *****

% Map to representation used in book
obs(F, true, I) :- obs(F, I).

% Assume correct initial state
holds(F, 0) :- obs(F, true, 0).

% Defined fluents are hardcoded, so we need to copy them over to holds
holds(F, I) :-
  fluent(defined, F),
  obs(F, true, I).

% CWA for obs
obs(F, false, I) :-
  fluent(_, F),
  not obs(F, true, I),
  step(I).

% Reality check axioms
:- obs(F, true, I), -holds(F, I).
% :- obs(F, false, I), holds(F, I).

% Generate possible events
occurs(A, I) : action(A) :- step(I+1), I >= 0.

% wrong_obs_true(F, I) :- obs(F, true, I), -holds(F, I).
% wrong_obs_false(F, I) :- obs(F, false, I), holds(F, I).
% #show wrong_obs_true/2.
% #show wrong_obs_false/2.

% correct_actions :- occurs(move(0), 0), occurs(rotate_right(0), 1).
% :- not correct_actions.

#show occurs/2.
