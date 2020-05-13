% *** Background knowledge ***

holds(F, I) :- obs(F, I).

step(I) :- obs(_, I).

obj_pos((X, Y), Id, Frame) :- holds(position((X, Y, _, _), Id), Frame).

disappear(Id, Frame+1) :-
  holds(class(Class, Id), Frame),
  not holds(class(Class, Id), Frame+1),
  step(Frame+1),
  step(Frame).

move_direction(left, Id, I) :- obj_pos((X1, Y1), Id, I), obj_pos((X2, Y2), Id, I+1),


% *** Language bias for move ***

% #modeh(occurs(move(var(id)), var(frame))).
%
% #modeb(1, holds(rotation(rot, var(id)), var(frame))).
% #modeb(1, obj_pos((var(x), var(y)), var(id), var(frame))).
% #modeb(1, obj_pos((var(next_x), var(next_y)), var(id), var(frame)+1)).
% #modeb(1, var(x)>var(next_x)).
% #modeb(1, var(x)<var(next_x)).
% #modeb(1, var(y)>var(next_y)).
% #modeb(1, var(y)<var(next_y)).
%
% #constant(rot, 0).
% #constant(rot, 1).
% #constant(rot, 2).
% #constant(rot, 3).
%
% #maxv(6).
% #max_penalty(10).

% combinations(A, B) = {A>B, A<B, A==B, true}

1 ~ occurs(move(Id), initial_frame) :-
  static(Id, initial_frame, false),
  obj_pos((X1, Y1), Id, initial_frame), obj_pos((X2, Y2), Id, next_frame),
  holds(rotation(0, Id), initial_frame), holds(rotation(0, Id), next_frame),
  holds(colour(blue, Id), initial_frame), holds(colour(blue, Id), next_frame),
  X1>X2, Y1>Y2.


#pos(p1, {
occurs(move(0), initial_frame).
}, {

}, {
obs(position((20,30,100,110), 0), initial_frame).
obs(position((20,50,100,130), 0), next_frame).
}).


% Try to learn model

% 1 ~ holds(position((X1, Y1, X2, Y2), Id), I) :-
%   occurs(move(Id), I),
%   holds(position((X1, Y1, X2, Y2), Id), I),
%   step(I+1).
