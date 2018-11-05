function [x, varargout] = fista(grad_f, prox_g, x0, L, varargin) 
%
% Solves 
%
% min_x F(x) := f(x) + g(x)
%
% for f differentiable with L-Lipschitz derivative 
% and f or g strongly convex with constant mu_f, mu_g using FISTA.
%
% Input: 
% 
% grad_f     == gradient of the function to be minimized (function handle)
% prox_g     == proximal operator for the function g (function handle)
% x0         == starting point 
% L          == Lipschitz constant of 'grad_f' 
%
% varargin   == niter: number of iterations 
%               mu_f : strong convexity constant of f (else = 0) 
%               mu_g : strong convexity constant of g (else = 0) 
%               F    : function to be minimized (function handle) 
%                      to monitor the energy
%               int  : interval, in which the energy is checked
%
% Output: 
%
% x          == solution of the problem
% hist       == history of the energy of the problem
%

% Defaults 
niter   = 500; 
mu_f    = 0;
mu_g    = 0;
F       = @(x) pi;
int     = 25;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Check, whether F is available
if (F(x0) ~= pi)
    fun_val = true;
else
    fun_val = false;
end

% Check whether history is requested
if (nargout - 1 == 1)
    if (fun_val)
        hist = zeros(niter/int,1);
    else 
        error('Please provide the function F.');
    end
end

tau = 0.99/L;
mu  = mu_f + mu_g;
q   = (tau * mu) / (1 + tau * mu_g);

% Initialize 
t_old = 0; 
x_old = x0; 
x     = x0;

for i = 1:niter 
    
    t = (1 - q * t_old^2 + sqrt((1 - q * t_old^2)^2 + 4 * t_old^2)) / 2;
    b = (t_old - 1)/t * (1 + tau * mu_g - t * tau * mu) / (1 - tau * mu_f);
    y = x + b * (x - x_old); 
    
    x_old = x;
    t_old = t; 

    x = prox_g(y - tau * grad_f(y),tau);
    
    if (fun_val && mod(i,int) == 0)
        hist(i/int) = F(x);
        fprintf('It:%d. Energy: %6.6d.\n',i, hist(i/int));
    end
    
end

if (nargout - 1 == 1)
    varargout{1} = hist;
end

end

