
% Choice rules
1 { colour_mapping(4, green) ; colour_mapping(4, silver) } 1.
1 { rotation_mapping(4, up) ; rotation_mapping(4, down) } 1.

1 { colour_mapping(7, green) ; colour_mapping(7, silver) } 1.
1 { rotation_mapping(7, up) ; rotation_mapping(7, down) } 1.


% Mapping to holds from <prop>_mapping
holds(colour(Val, Id), Q) :- labelled_obj(Id, Label, Q), colour_mapping(Label, Val).
holds(rotation(Val, Id), Q) :- labelled_obj(Id, Label, Q), rotation_mapping(Label, Val).

% mapping(Label, Col, Rot) :- colour_mapping(Label, Col), rotation_mapping(Label, Rot).
% :- mapping(Label1, Col, Rot), mapping(Label2, Col, Rot), Label1 != Label2.


% QA data
answer(12, colour, V) :- holds(rotation(up, Id), 12), holds(colour(V, Id), 12).
expected(12, colour, silver).

answer(12, rotation, V) :- holds(colour(silver, Id), 12), holds(rotation(V, Id), 12).
expected(12, rotation, up).


% Frame data
labelled_obj(0, 4, 12).
labelled_obj(1, 7, 12).

% Optimisation
:~ answer(Q, Prop, Val), expected(Q, Prop, Val). [-1@1, Q, Prop, Val]

#show colour_mapping/2.
#show rotation_mapping/2.
