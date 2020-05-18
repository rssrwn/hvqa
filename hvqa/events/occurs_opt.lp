holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).

object(Id, Frame) :- holds(class(_, Id), Frame).

disappear(Id, -I) :-
  holds(class(Class, Id), I),
  not holds(class(Class, Id), -I),
  step(-I), step(I).

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


% Weak constraints
:~ predicted(action, Frame), actual(action, Frame). [-1@3, Frame]
:~ actual(Action, Frame), Action != action, not predicted(action, Frame). [-1@3, Frame]
:~ count_rules(Cnt). [Cnt@1]
:~ Sum = #sum{ Size, R : rule_size(Size, R), rule(R) }. [Sum@0]

#show feature/3.
#show feature/3.
