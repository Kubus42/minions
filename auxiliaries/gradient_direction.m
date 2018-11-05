function gradx = gradient_direction(x,varargin)

% This function creates an normalized gradient 'p' from the image 'x': 
% 
% p = gradx/|gradx|.
%
% The threshold parameter 'eta' decides, how large gradients
% have to be to be considered as an edge. 
%
% Input: 
% x        ==   Image, for which the subgradient is computed
% varargin ==   eta: The edge parameter. 
%
% Questions to julian.rasch@wwu.de

% Overload varargin
% Defaults: 
eta = 0.01;

if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Compute the gradient of x 
gradx = grad(x); 
dim   = ndims(gradx);
sz    = size(gradx);

% Compute its norm 
n_gradx = repmat(sqrt(sum(abs(gradx).^2,dim)),[1,1,sz(end)]);

% Compute the normalized gradient 
gradx = gradx ./ n_gradx; 

max_gradx = max(n_gradx(:));

% Set gradients smaller than eta to zero
ind = (n_gradx < eta * max_gradx);
gradx(ind) = 0;

% Check the result
% "Edges"
% figure; imagesc(sqrt(abs(gradx(:,:,1)).^2 + abs(gradx(:,:,2)).^2)); colorbar;
% "Subgradient"
% figure; imagesc(abs(p)); colorbar; 



