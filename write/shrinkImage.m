function [img, num] = shrinkImage(img,a,b)
%SHRINK IMAGE between two values
%
% Shrinks the values of the input image 'img' to the range [a b].
%
% Input:    
%   img [matrix]              
%         scalar valued image
%
%   a   [float; optional]              
%         lower bound for projection. DEFAULT = 0
%
%   b   [float; optional]              
%         upper bound for projection. DEFAULT = 1    
%
% Output: 
%   img [matrix] 
%         the shrunk image 
%   
%   num  [int]
%         number of shrunk values
 
if nargin < 2
    a = 0;
    b = 1;
end

l = find(img<a);
u = find(img>b);

num = (numel(l) + numel(u)) / numel(img);
img(l) = a; 
img(u) = b;

end