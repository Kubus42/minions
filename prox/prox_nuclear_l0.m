function X = prox_nuclear_l0(Y, alpha)
% Proximal operator for the nuclear norm of a matrix Y, i.e. 
% singular value hard thresholding.
%
% Solves 
%
% min_X 0.5 * |X - Y|^2 s.t. rk(X) = alpha
%
% Input: 
% Y          ==   data
% alpha      ==   rank

[U,S,V] = svd(Y);

s = diag(S);
s(alpha+1:end) = 0;
SS = diag(s);

X = U * SS * V';

end