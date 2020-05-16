
#const action=rotate_right.
#const max_rules=5.
#const fg=disappear.


holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).

object(Id, Frame) :- holds(class(_, Id), Frame).

disappear(Id, -I) :-
  holds(class(Class, Id), I),
  not holds(class(Class, Id), -I),
  step(-I), step(I).

static(Id, I, false) :- holds(class(octopus, Id), I).
static(Id, I, true) :- holds(class(fish, Id), I).
static(Id, I, true) :- holds(class(rock, Id), I).
static(Id, I, true) :- holds(class(bag, Id), I).


rule_size(Size, R) :- Size = #sum{ W, Fg, F_id : feature_weight(Fg, F_id, W), feature(Fg, F_id, R) }, rule(R).

count_rules(Cnt) :- Cnt = #count{ R : rule(R) }.


1 { max_rule_id(0..max_rules-1) } 1.
rule(0..Max) :- max_rule_id(Max).


predicted(action, Frame) :-
  static(Id, Frame, false),
  acc_value(Id, Frame, Rule),
  feature_value(fg, Id, Frame, Rule),
  rule(Rule).


% We can choose to either use all these features or none, for each rule in H
1 { use_acc(Acc_id, Rule) : acc_id(Acc_id) ; no_acc(Rule) } 1 :- rule(Rule).


acc_value(Id, Frame, Rule) :-
  feature_value(x_pos, Id, Frame, Rule),
  feature_value(y_pos, Id, Frame, Rule),
  feature_value(rot, Id, Frame, Rule),
  feature_value(colour, Id, Frame, Rule),
  feature_value(disappear, Id, Frame, Rule),
  static(Id, Frame, false).

% Accumulated features
acc_feature(x_pos, 4, 0).
acc_feature(x_pos, 4, 1).
acc_feature(x_pos, 4, 2).
acc_feature(x_pos, 4, 3).
acc_feature(y_pos, 4, 0).
acc_feature(y_pos, 4, 1).
acc_feature(y_pos, 4, 2).
acc_feature(y_pos, 4, 3).
acc_feature(rot, 1, 3).
acc_feature(rot, 6, 2).
acc_feature(rot, 12, 1).
acc_feature(rot, 11, 0).
acc_feature(colour, 51, 0).
acc_feature(colour, 51, 1).
acc_feature(colour, 51, 2).
acc_feature(colour, 51, 3).

acc_id(Acc_id) :- acc_feature(_, _, Acc_id).

feature(Fg, F_id, Rule) :-
  use_acc(Acc_id, Rule),
  acc_feature(Fg, F_id, Acc_id).

acc_fg(Fg) :- acc_feature(Fg, _, _).
feature(Fg, Empty_id, Rule) :-
  acc_fg(Fg),
  empty_id(Fg, Empty_id),
  no_acc(Rule).

% Set any feature not yet being optimised to empty
feature(Fg, Empty_id, Rule) :-
  not acc_fg(Fg),
  Fg != fg,
  empty_id(Fg, Empty_id),
  rule(Rule).


% Generate all possible feature combinations
1 { feature(fg, F_id, R) : feature_weight(fg, F_id, _) } 1 :- rule(R).


% X position features
feature_value(x_pos, Id, Frame, Rule) :-
  X1<X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 0, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1>X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 1, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1=X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 2, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1!=X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 3, Rule).

feature_value(x_pos, Id, Frame, Rule) :- object(Id, Frame), feature(x_pos, 4, Rule).

feature_weight(x_pos, 0, 1).
feature_weight(x_pos, 1, 1).
feature_weight(x_pos, 2, 1).
feature_weight(x_pos, 3, 1).
feature_weight(x_pos, 4, 0).

empty_id(x_pos, 4).


% Y position features
feature_value(y_pos, Id, Frame, Rule) :-
  Y1<Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 0, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1>Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 1, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1=Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 2, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1!=Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 3, Rule).

feature_value(y_pos, Id, Frame, Rule) :- object(Id, Frame), feature(y_pos, 4, Rule).

feature_weight(y_pos, 0, 1).
feature_weight(y_pos, 1, 1).
feature_weight(y_pos, 2, 1).
feature_weight(y_pos, 3, 1).
feature_weight(y_pos, 4, 0).

empty_id(y_pos, 4).


% Rotation features
feature_value(rot, Id, Frame, Rule) :-
  R1=0, R2=0, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 0, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=0, R2=1, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 1, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=0, R2=2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 2, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=0, R2=3, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 3, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=1, R2=0, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 4, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=1, R2=1, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 5, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=1, R2=2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 6, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=1, R2=3, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 7, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=2, R2=0, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 8, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=2, R2=1, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 9, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=2, R2=2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 10, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=2, R2=3, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 11, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=3, R2=0, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 12, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=3, R2=1, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 13, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=3, R2=2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 14, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=3, R2=3, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 15, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1=R2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 16, Rule).

feature_value(rot, Id, Frame, Rule) :-
  R1!=R2, holds(rotation(R1, Id), Frame), holds(rotation(R2, Id), -Frame), feature(rot, 17, Rule).

feature_value(rot, Id, Frame, Rule) :- object(Id, Frame), feature(rot, 18, Rule).

feature_weight(rot, 0, 1).
feature_weight(rot, 1, 1).
feature_weight(rot, 2, 1).
feature_weight(rot, 3, 1).
feature_weight(rot, 4, 1).
feature_weight(rot, 5, 1).
feature_weight(rot, 6, 1).
feature_weight(rot, 7, 1).
feature_weight(rot, 8, 1).
feature_weight(rot, 9, 1).
feature_weight(rot, 10, 1).
feature_weight(rot, 11, 1).
feature_weight(rot, 12, 1).
feature_weight(rot, 13, 1).
feature_weight(rot, 14, 1).
feature_weight(rot, 15, 1).
feature_weight(rot, 16, 1).
feature_weight(rot, 17, 1).
feature_weight(rot, 18, 0).

empty_id(rot, 18).


% Colour features
feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 0, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 1, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 2, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 3, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 4, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 5, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=0, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 6, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 7, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 8, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 9, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 10, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 11, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 12, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=1, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 13, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 14, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 15, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 16, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 17, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 18, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 19, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=2, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 20, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 21, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 22, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 23, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 24, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 25, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 26, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=3, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 27, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 28, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 29, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 30, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 31, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 32, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 33, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=4, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 34, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 35, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 36, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 37, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 38, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 39, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 40, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=5, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 41, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=0, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 42, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=1, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 43, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 44, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=3, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 45, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=4, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 46, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=5, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 47, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1=6, C2=6, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 48, Rule).


feature_value(colour, Id, Frame, Rule) :-
  C1=C2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 49, Rule).

feature_value(colour, Id, Frame, Rule) :-
  C1!=C2, holds(colour(C1, Id), Frame), holds(colour(C2, Id), -Frame), feature(colour, 50, Rule).

feature_value(colour, Id, Frame, Rule) :- object(Id, Frame), feature(colour, 51, Rule).

feature_weight(colour, 0, 1).
feature_weight(colour, 1, 1).
feature_weight(colour, 2, 1).
feature_weight(colour, 3, 1).
feature_weight(colour, 4, 1).
feature_weight(colour, 5, 1).
feature_weight(colour, 6, 1).
feature_weight(colour, 7, 1).
feature_weight(colour, 8, 1).
feature_weight(colour, 9, 1).
feature_weight(colour, 10, 1).
feature_weight(colour, 11, 1).
feature_weight(colour, 12, 1).
feature_weight(colour, 13, 1).
feature_weight(colour, 14, 1).
feature_weight(colour, 15, 1).
feature_weight(colour, 16, 1).
feature_weight(colour, 17, 1).
feature_weight(colour, 18, 1).
feature_weight(colour, 19, 1).
feature_weight(colour, 20, 1).
feature_weight(colour, 21, 1).
feature_weight(colour, 22, 1).
feature_weight(colour, 23, 1).
feature_weight(colour, 24, 1).
feature_weight(colour, 25, 1).
feature_weight(colour, 26, 1).
feature_weight(colour, 27, 1).
feature_weight(colour, 28, 1).
feature_weight(colour, 29, 1).
feature_weight(colour, 30, 1).
feature_weight(colour, 31, 1).
feature_weight(colour, 32, 1).
feature_weight(colour, 33, 1).
feature_weight(colour, 34, 1).
feature_weight(colour, 35, 1).
feature_weight(colour, 36, 1).
feature_weight(colour, 37, 1).
feature_weight(colour, 38, 1).
feature_weight(colour, 39, 1).
feature_weight(colour, 40, 1).
feature_weight(colour, 41, 1).
feature_weight(colour, 42, 1).
feature_weight(colour, 43, 1).
feature_weight(colour, 44, 1).
feature_weight(colour, 45, 1).
feature_weight(colour, 46, 1).
feature_weight(colour, 47, 1).
feature_weight(colour, 48, 1).
feature_weight(colour, 49, 1).
feature_weight(colour, 50, 1).
feature_weight(colour, 51, 0).

empty_id(colour, 51).


% Disappear features
feature_value(disappear, Id, Frame, Rule) :- disappear(Id, -Frame), feature(disappear, 0, Rule).
feature_value(disappear, Id, Frame, Rule) :- object(Id, Frame), not disappear(Id, -Frame), feature(disappear, 1, Rule).
feature_value(disappear, Id, Frame, Rule) :- object(Id, Frame), feature(disappear, 2, Rule).

feature_weight(disappear, 0, 1).
feature_weight(disappear, 1, 1).
feature_weight(disappear, 2, 0).

empty_id(disappear, 2).


% Weak constraints
:~ predicted(action, Frame), actual(action, Frame). [-1@3, Frame]
:~ actual(Action, Frame), Action != action, not predicted(action, Frame). [-1@3, Frame]
:~ count_rules(Cnt). [Cnt@1]
:~ Sum = #sum{ Size, R : rule_size(Size, R), rule(R) }. [Sum@0]

#show feature/3.
#show feature/3.
