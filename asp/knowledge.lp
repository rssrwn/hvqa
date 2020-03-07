% object(Id, Class, Colour, Rot, Pos, Frame) :- property(class, Class, Id, Frame),
%                                               property(colour, Colour, Id, Frame),
%                                               property(rotation, Rot, Id, Frame),
%                                               property(position, Pos, Id, Frame).

exists(Id, Frame) :- object(Id, _, _, _, _, Frame).

changed(P, From, To, Id, Frame) :- property(P, From, Id, Frame), property(P, To, Id, Frame+1), From != To, P = class.
changed(P, From, To, Id, Frame) :- property(P, From, Id, Frame), property(P, To, Id, Frame+1), From != To, P = colour.
changed(P, From, To, Id, Frame) :- property(P, From, Id, Frame), property(P, To, Id, Frame+1), From != To, P = rotation.
changed(P, From, To, Id, Frame) :- property(P, From, Id, Frame), property(P, To, Id, Frame+1), From != To, P = position.
