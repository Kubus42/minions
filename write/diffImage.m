function d = diffImage(x,y,s)
%DIFFERENCE IMAGE 
%
% Computes the difference between the image x and and a ground truth
% image y and puts it onto a unit scale between [-0.5, 0.5]. 
% It shrinks all deviations above 50 percent. 
%
% Input:    
%   x [matrix]              
%        scalar valued image
%
%   y [matrix]              
%        scalar valued image
%
%   s [matrix]  [s1,s2]             
%        scale of the image y

d = x - y;
k = (s(2)-s(1))/2;
d = shrinkImage(d,-k,k);
d = d/(2*k);

end