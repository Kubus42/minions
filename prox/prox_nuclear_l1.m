function [X,nn] = prox_nuclear_l1(Y, alpha)
% Proximal operator for the nuclear norm of a matrix Y, i.e. 
% singular value soft thresholding.
%
% Solves 
%
% min_X 0.5 * |X - Y|^2 + alpha * |X|_* 
%
% Input: 
% Y          ==   data
% alpha      ==   regularization parameter

[U,S,V] = svd(Y,0);

SS = max(0, 1-alpha./max(S,1e-9)) .* S;
nn = sum(diag(SS));

X = U * SS * V';

end