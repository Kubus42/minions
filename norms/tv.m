function nn = tv(f, varargin)
%(AN)ISOTROPIC (HUBER) TOTAL VARIATION
% Computes 
%
% TV(f) = | ||grad f|| |_1, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Requires: grad
%
% Input: 
% f          ==   data
% varargin   ==   'norm_type': 'aniso' / 'iso'
%                 epsilon: The Huber epsilon
%
% Output:
% nn         ==   the total variation of 'f'

grad_f  = grad(f);
sz      = size(grad_f);

% Helper 
vec = @(x) x(:);

% Anisotropic 
if (isempty(varargin) == 1 || strcmp(varargin{1}, 'aniso'))
    % if epsilon > 0, compute Huber TV, else standard TV
    if (numel(varargin) > 1 && varargin{2} > 0)
        epsilon = varargin{2};
        aux = abs(vec(grad_f)); 
        nn = sum(aux(aux>epsilon) - epsilon/2);
        nn = nn + sum((aux(aux<=epsilon).^2)/(2*epsilon));
    else
        nn = norm(vec(grad_f),1); 
    end
    
    % Isotropic
elseif (strcmp(varargin{1}, 'iso') && numel(sz(sz~=1)) > 1 )
    % if epsilon > 0, compute Huber TV, else standard TV
    if (numel(varargin) > 1 && varargin{2} > 0)
        epsilon = varargin{2};
        aux = reshape(grad_f,[prod(sz(1:end-1)),sz(end)]);
        aux = sqrt(sum(aux.^2,2));
        nn = sum(aux(aux>epsilon) - epsilon/2);
        nn = nn + sum((aux(aux<=epsilon).^2)/(2*epsilon)); 
    else
        nn   = reshape(grad_f,[prod(sz(1:end-1)),sz(end)]);
        nn   = sum(vec(sqrt(sum(nn.^2,2))));
    end
else 
    error('Check input!');
end
end
