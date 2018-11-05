function divx = divx(I)
%DIVERGENCE IN X-DIRECTION of a 2D image I
%
% only works in 2D, to be used with 'symdiv', 'symgrad'

s  = size(I);
n = s(1);

%Divergence
tx = [n 1:n-1];
divx         = I(:, :, 1) - I(tx, :, 1);
divx(1, :)   = I(1, :, 1);
divx(end, :) = -I(end-1, :, 1);
