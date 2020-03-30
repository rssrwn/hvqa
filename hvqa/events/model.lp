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
#const n = 4.
step(3..n).

#const num_objects = 3.
id(0..(num_objects-1)).

% Type definitions
fluent(inertial, rotation(R, Id)) :- holds(rotation(R, Id), _).
fluent(inertial, position(P, Id)) :- holds(position(P, Id), _).
fluent(inertial, class(C, Id)) :- holds(class(C, Id), _).
fluent(inertial, colour(C, Id)) :- holds(colour(C, Id), _).

fluent(defined, close(Id1, Id2)) :- holds(close(Id1, Id2), _).

% Assume only octopus can do actions
action(move(Id)) :- holds(class(octopus, Id), _).
action(rotate_left(Id)) :- holds(class(octopus, Id), _).
action(rotate_right(Id)) :- holds(class(octopus, Id), _).


% ***** Domain Independent Axioms *****

% CWA for defined fluents
-holds(F, I) :-
  fluent(defined, F),
  not holds(F, I),
  step(I).

% CWA for actions
-occurs(A, I) :-
  not occurs(A, I),
  action(A),
  step(I).

% Inertia axioms
holds(F, I+1) :-
  fluent(inertial, F),
  holds(F, I),
  not -holds(F, I+1),
  step(I+1).

-holds(F, I+1) :-
  fluent(inertial, F),
  -holds(F, I),
  not holds(F, I+1),
  step(I+1),
  step(I).


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

% Definition for rotate
holds(rotation(@new_rot(Rot, "rotate_left"), I+1)) :-
  occurs(rotate_left(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

-holds(rotation(Rot, I+1)) :-
  occurs(rotate_left(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

holds(rotation(@new_rot(Rot, "rotate_right"), I+1)) :-
  occurs(rotate_right(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

-holds(rotation(Rot, I+1)) :-
  occurs(rotate_right(Id), I),
  holds(rotation(Rot, Id), I),
  step(I+1).

% Definition for colour
holds(colour(Col, Id), I) :-
  holds(class(octopus, Id), I),
  holds(close(Id, IdRock), I),
  holds(class(rock, IdRock), I),
  holds(colour(Col, IdRock), I).
