function divy = divy(I)
%DIVERGENCE IN Y-DIRECTION of a 2D image I
%
% only works in 2D, to be used with 'symdiv', 'symgrad'
s  = size(I);
m = s(2);

ty = [m 1:m-1];
divy         = I(:, :, 1) - I(:,ty,1);
divy(:, 1)   = I(:, 1, 1);
divy(:, end) = -I(:, end-1, 1);


