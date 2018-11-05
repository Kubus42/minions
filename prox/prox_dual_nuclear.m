function X = prox_dual_nuclear(Y, t)

% Proximal operator for the dual of the nuclear norm of a matrix Y, i.e. 
% a projection of the singular values onto the t-ball.
%
% Solves 
%
% min_X 0.5 * |X - Y|_F^2 s.t. sigma_i(X) <= t for all i.
%
% Input: 
% Y      ==   data
% t      ==   regularization parameter

[U,S,V] = svd(Y,0);

SS = S./max(1,S./t);

X = U * SS * V';

end