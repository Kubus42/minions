function x = prox_l0(f, alpha, varargin)
% Proximal operator for the l0-norm, i.e. hard thresholding.
%
% Solves 
%
% min_x 0.5 * |x - f|^2 + alpha * | ||x|| |_0, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Input: 
% f          ==   data
% alpha      ==   regularization parameter
% varargin   ==   norm type 'aniso' / 'iso' 

sz = size(f);

% Anisotropic shrinkage
if (isempty(varargin) == 1 || strcmp(varargin{1}, 'aniso'))
    x = (abs(f) > alpha) .* f; 
% Isotropic shrinkage
elseif (strcmp(varargin{1}, 'iso') && numel(sz(sz~=1)) > 1 )
    x   = reshape(f,[prod(sz(1:end-1)),sz(end)]);
    aux = sqrt(sum(x.^2,2));
    for i = 1:sz(end)
        x(:,i) = (aux > alpha) .* x(:,i);
    end
    x   = reshape(x,sz);
else 
    error('Check input!');
end


end