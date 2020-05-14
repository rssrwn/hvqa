
holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).

disappear(Id, I+1) :-
  holds(class(Class, Id), I),
  not holds(class(Class, Id), I+1),
  step(I+1), step(I).

static(Id, I, false) :- holds(class(octopus, Id), I).
static(Id, I, true) :- holds(class(fish, Id), I).
static(Id, I, true) :- holds(class(rock, Id), I).
static(Id, I, true) :- holds(class(bag, Id), I).


rule_size(Size, R) :- Size = #sum{ W, Fg, F_id : feature_weight(Fg, F_id, W), feature(Fg, F_id, R) }, rule(R).

count_rules(Cnt) :- Cnt = #count{ R : rule(R) }.


max_rules(8).
1 { max_rule_id(0..Max-1) } 1 :- max_rules(Max).
rule(0..Max) :- max_rule_id(Max).


predicted(move, Frame) :-
  static(Id, Frame, false),
  feature_value(x_pos, Id, Frame, Rule),
  feature_value(y_pos, Id, Frame, Rule),
  rule(Rule).

% predicted(rotate_left, Frame) :-
%   static(Id, Frame, false),
%   feature_value(x_pos, Id, Frame, Rule),
%   feature_value(y_pos, Id, Frame, Rule),
%   rule(Rule).
%
% predicted(rotate_right, Frame) :-
%   static(Id, Frame, false),
%   feature_value(x_pos, Id, Frame, Rule),
%   feature_value(y_pos, Id, Frame, Rule),
%   rule(Rule).


% X position features
feature_value(x_pos, Id, Frame, Rule) :-
  X1<X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 0, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1>X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 1, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1=X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 2, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  X1!=X2, obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 3, Rule).

feature_value(x_pos, Id, Frame, Rule) :-
  obj_pos((X1, _), Id, Frame), obj_pos((X2, _), Id, -Frame), feature(x_pos, 4, Rule).

1{ feature(x_pos, 0, R) ; feature(x_pos, 1, R) ; feature(x_pos, 2, R) ; feature(x_pos, 3, R) ; feature(x_pos, 4, R) }1 :- rule(R).

feature_weight(x_pos, 0, 1).
feature_weight(x_pos, 1, 1).
feature_weight(x_pos, 2, 1).
feature_weight(x_pos, 3, 1).
feature_weight(x_pos, 4, 0).


% Y position features
feature_value(y_pos, Id, Frame, Rule) :-
  Y1<Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 0, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1>Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 1, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1=Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 2, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  Y1!=Y2, obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 3, Rule).

feature_value(y_pos, Id, Frame, Rule) :-
  obj_pos((_, Y1), Id, Frame), obj_pos((_, Y2), Id, -Frame), feature(y_pos, 4, Rule).

1{ feature(y_pos, 0, R) ; feature(y_pos, 1, R) ; feature(y_pos, 2, R) ; feature(y_pos, 3, R) ; feature(y_pos, 4, R) }1 :- rule(R).

feature_weight(y_pos, 0, 1).
feature_weight(y_pos, 1, 1).
feature_weight(y_pos, 2, 1).
feature_weight(y_pos, 3, 1).
feature_weight(y_pos, 4, 0).


% Weak constraints
:~ predicted(Action, Frame), actual(Action, Frame). [-1@3, Action, Frame]
:~ predicted(Pred, Frame), actual(Act, Frame), Pred != Act. [1@2, Pred, Act, Frame]
:~ count_rules(Cnt). [Cnt@1]
:~ Sum = #sum{ Size, R : rule_size(Size, R), rule(R) }. [Sum@0]

% :- predicted(Action1, Frame), predicted(Action2, Frame), Action1 != Action2.

#show feature/3.
#show feature/3.
