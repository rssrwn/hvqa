#modeh(-holds_next(position((var(x1), var(y1), var(x2), var(y2)), var(id)))).
#modeh(holds_next(position((var(x1) + const(add_x), var(y1), var(x2) + const(add_x), var(y2)), var(id)))).
#modeh(holds_next(position((var(x1), var(y1) + const(add_y), var(x2), var(y2) + const(add_y)), var(id)))).
#modeh(holds_next(position((var(x1) + const(add_x), var(y1) + const(add_y), var(x2) + const(add_x), var(y2) + const(add_y)), var(id)))).

#modeh(holds_next(rotation(var(rot) - 1, var(id)))).
% #modeh(-holds_next(rotation(var(rot), var(id)))).

#modeb(1, occurs(move(var(id))), (positive)).
#modeb(1, occurs(rotate_left(var(id))), (positive)).
#modeb(1, holds_curr(position((var(x1), var(y1), var(x2), var(y2)), var(id)))).
#modeb(1, holds_curr(rotation(var(rot), var(id)))).

#maxv(15).

#constant(add_x, 5).
#constant(add_x, 10).
#constant(add_x, 15).
#constant(add_y, 5).
#constant(add_y, 10).
#constant(add_y, 15).

% Inertia axioms
holds_next(F) :-
  holds_curr(F),
  not -holds_next(F).

% -holds_next(F) :-
%   -holds_curr(F, I),
%   not holds_next(F).

#pos(p1, {
  holds_next(position((25,60,45,70), 0)),
  -holds_next(position((10,60,30,70), 0)),
  holds_next(rotation(1, 0))
}, {}, {
  holds_curr(position((10,60,30,70), 0)).
  holds_curr(rotation(1, 0)).
  occurs(move(0)).
}).

#pos(p2, {
  holds_next(position((25,60,45,70), 0)),
  holds_next(rotation(0, 0))
}, {}, {
  holds_curr(position((25,60,45,70), 0)).
  holds_curr(rotation(1, 0)).
  occurs(rotate_left(0)).
}).
