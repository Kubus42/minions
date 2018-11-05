function nn = dtv(f, v, varargin)
%(HUBER) DIRECTIONAL TOTAL VARIATION
%
% Computes | || P_v^orth (grad f)|| |_1, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Input: 
% f          ==   data
% v          ==   vector field for the projection
% varargin   ==   'norm_type': 'aniso' / 'iso' 
%
% Output: 
% nn         ==   the directional total variation of 'f'

grad_f  = grad(f);
sz      = size(grad_f);
dim     = numel(sz);

if (size(v) ~= sz) 
    error('Dimension mismatch between grad(f) and vector field v.');
end

% Compute the projection
grad_f = grad_f - sum(grad_f .* v,dim) .* v;

% Helper 
vec = @(x) x(:);

% Anisotropic 
if (isempty(varargin) == 1 || strcmp(varargin{1}, 'aniso'))
    % if epsilon > 0 compute Huber dTV, else standard dTV
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
    % if epsilon > 0 compute Huber dTV, else standard dTV
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
