% Python functions

#script (python)

import clingo
F = clingo.Function


def new_pos(pos, rotation):
    rotation = rotation.number
    x1, y1, x2, y2 = pos.arguments
    x1 = x1.number
    y1 = y1.number
    x2 = x2.number
    y2 = y2.number
    if rotation == 0:
        pos = F("", (x1, y1-15, x2, y2-15))
    elif rotation == 1:
        pos = F("", (x1+15, y1, x2+15, y2))
    elif rotation == 2:
        pos = F("", (x1, y1+15, x2, y2+15))
    elif rotation == 3:
        pos = F("", (x1-15, y1, x2-15, y2))

    return pos


def new_rot(rotation, action):
    rotation = rotation.number
    action = action.string
    if action == "rotate_left":
        rot = rotation - 1
        if rot < 0:
            rot = 3
    elif action == "rotate_right":
        rot = rotation + 1
        if rot > 3:
            rot = 0

    return rot

#end.


% ***** Types *****

% Timesteps
% Assume every we have information for every timestep
step(I) :- obs(_, _, I).

% Add internal property called exists
obs(exists(Id), true, I) :- obs(class(_, Id), true, I).

% Type definitions
fluent(inertial, rotation(R, Id)) :- obs(rotation(R, Id), _, _).
fluent(inertial, position(P, Id)) :- obs(position(P, Id), _, _).
fluent(inertial, class(C, Id)) :- obs(class(C, Id), _, _).
fluent(inertial, colour(C, Id)) :- obs(colour(C, Id), _, _).
fluent(inertial, exists(Id)) :- obs(exists(Id), _, _).

fluent(defined, close(Id1, Id2)) :- obs(close(Id1, Id2), _, _).

% Assume only octopus can do actions
action(move(Id)) :- obs(class(octopus, Id), _, _).
action(rotate_left(Id)) :- obs(class(octopus, Id), _, _).
action(rotate_right(Id)) :- obs(class(octopus, Id), _, _).
action(nothing(Id)) :- obs(class(octopus, Id), _, _).


% ***** Domain Independent Axioms *****

% CWA for inertial fluents (initial frame only)
-holds(F, 0) :-
  fluent(inertial, F),
  not holds(F, 0).

% CWA for inertial fluents
% -holds(F, I) :-
%   fluent(inertial, F),
%   not holds(F, I),
%   step(I).

% CWA for defined fluents
-holds(F, I) :-
  fluent(defined, F),
  not holds(F, I),
  step(I).

% CWA for actions
-occurs(A, I) :-
  not occurs(A, I),
  action(A),
  step(I+1),
  step(I).

% Inertia axioms
holds(F, I+1) :-
  fluent(inertial, F),
  holds(F, I),
  not -holds(F, I+1),
  step(I+1),
  step(I).

-holds(F, I+1) :-
  fluent(inertial, F),
  -holds(F, I),
  not holds(F, I+1),
  step(I+1),
  step(I).

% Assume only one object can do actions
non_static(Id) :- occurs(move(Id), Frame).
non_static(Id) :- occurs(rotate_left(Id), Frame).
non_static(Id) :- occurs(rotate_right(Id), Frame).
non_static(Id) :- occurs(nothing(Id), Frame).

:- non_static(Id1), non_static(Id2), Id1 != Id2.


% ***** Domain Dependent *****

% Definition for move
holds(position(@new_pos(Pos, Rot), Id), I+1) :-
  occurs(move(Id), I),
  holds(position(Pos, Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

-holds(position(Pos, Id), I+1) :-
  occurs(move(Id), I),
  holds(position(Pos, Id), I),
  step(I+1).

% Definition for rotate left
holds(rotation(@new_rot(Rot, "rotate_left"), Id), I+1) :-
  occurs(rotate_left(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

-holds(rotation(Rot, Id), I+1) :-
  occurs(rotate_left(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

% Definition for rotate right
holds(rotation(@new_rot(Rot, "rotate_right"), Id), I+1) :-
  occurs(rotate_right(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

-holds(rotation(Rot, Id), I+1) :-
  occurs(rotate_right(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

% Definition for colour
change_colour(Old, New, Id, I-1) :-
  holds(class(octopus, Id), I),
  holds(close(Id, IdRock), I),
  holds(colour(Old, Id), I-1),
  holds(class(rock, IdRock), I),
  holds(colour(New, IdRock), I),
  Old != New,
  step(I-1).

holds(colour(New, Id), I+1) :- change_colour(_, New, Id, I).
-holds(colour(Old, Id), I+1) :- change_colour(Old, _, Id, I).


% Event definitions
event(eat_fish(Octo), Frame) :-
  obs_disappear(Fish, Frame),
  holds(class(fish, Fish), Frame),
  occurs(move(Octo), Frame).

event(eat_bag(Octo), Frame) :-
  obs_disappear(Bag, Frame),
  holds(class(bag, Bag), Frame),
  occurs(move(Octo), Frame).

% Definition for exists
-holds(exists(Id), I+1) :-
  holds(exists(Id), I),
  obs_disappear(Id, I),
  step(I+1).

-occurs(move(Id), I) :- -holds(exists(Id), I).
-occurs(rotate_left(Id), I) :- -holds(exists(Id), I).
-occurs(rotate_right(Id), I) :- -holds(exists(Id), I).
