function x = prox_dual_l1(f, alpha, varargin)

% Proximal operator for the dual of the l1-norm, i.e. a pointwise 
% projection onto the alpha-ball.
%
% Solves 
%
% min_x 0.5 * |x - f|^2 s.t. ||x_i|| <= alpha for all i. 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Input: 
% f          ==   data
% alpha      ==   size of the ball
% varargin   ==   norm type 'aniso' / 'iso' 

sz = size(f); 

% Anisotropic case
if (isempty(varargin) == 1 || strcmp(varargin{1}, 'aniso'))
    x = f./max(1,abs(f)/alpha);
% Isotropic case
elseif (strcmp(varargin{1}, 'iso') && numel(sz(sz~=1)) > 1 )
    x   = reshape(f,[prod(sz(1:end-1)),sz(end)]);
    aux = sqrt(sum(x.^2,2));
    for i = 1:sz(end)
        x(:,i) = x(:,i)./max(1,aux/alpha);
    end
    x   = reshape(x,sz);
else 
    error('Check input!');
end
    
end