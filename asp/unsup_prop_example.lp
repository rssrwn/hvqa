
% Choice rules

% Label 0
1{ colour_mapping(0, green) ; colour_mapping(0, blue) ; ... }1. % repeated for all colours
1{ rotation_mapping(0, up) ; rotation_mapping(0, left) ; ... }1. % repeated for all rotations

% Label 1
1{ colour_mapping(1, green) ; colour_mapping(1, blue) ; ... }1. % repeated for all colours
1{ rotation_mapping(1, up) ; rotation_mapping(1, left) ; ... }1. % repeated for all rotations

% ... repeated for all labels


% Helper rules
obj_id(Id, Q) :- labelled_obj(Id, _, Q).
mapping(Label, Col, Rot) :- colour_mapping(Label, Col), rotation_mapping(Label, Rot).


% Mapping to holds from <prop>_mapping
holds(colour(Val, Id), Q) :- labelled_obj(Id, Label, Q), colour_mapping(Label, Val).
holds(rotation(Val, Id), Q) :- labelled_obj(Id, Label, Q), rotation_mapping(Label, Val).


% QA data

% Note: There is no data on class since, for each question, we only keep objects whose class corresponds to the class mentioned in the question

% Q: What is the colour of the upward-facing fish? A: silver
answer(0, colour, V) :- holds(rotation(up, Id), 0), holds(colour(V, Id), 0), obj_id(Id, 0).
expected(0, colour, silver).

answer(0, rotation, V) :- holds(colour(silver, Id), 0), holds(rotation(V, Id), 0), obj_id(Id, 0).
expected(0, rotation, up).


% Each class is run separately, this is just an example
% Q: What is the rotation of the octopus? A: down
% answer(1, rotation, V) :- holds(rotation(V, Id), 1), obj_id(Id, 1).
% expected(1, rotation, down).

% ... repeated for each question


% Frame data

% Note: labelled_obj(Id, Label, Question)

% Q1
labelled_obj(0, 3, 0).
labelled_obj(1, 2, 0).
labelled_obj(2, 2, 0).
labelled_obj(3, 0, 0).

% Q2
labelled_obj(0, 7, 1).

% ... repeated for each question


% Optimisation

% Choose the mapping which answers the most questions correctly
:~ answer(Q, Prop, Val), expected(Q, Prop, Val). [-1@1, Q, Prop, Val]

% Prefer answer sets where mappings are unqiue
:~ mapping(Label1, Col, Rot), mapping(Label2, Col, Rot), Label1 != Label2. [1@0, Label1, Label2, Col, Rot]


#show colour_mapping/2.
#show rotation_mapping/2.
