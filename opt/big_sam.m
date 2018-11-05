function [x, varargout] = big_sam(grad_w, grad_f, prox_g, x0, L_w, L_f, sigma, varargin) 
%
% Solves 
%
% min_x w(x) s.t. x \in arg min_x F(x) := f(x) + g(x),
%
% where w is differentiable with 'L_w'-Lipschitz derivative and strongly 
% convex with parameter 'sigma'. f is differentiable with L_f-Lipschitz 
% derivative.
%
% Input: 
% 
% grad_w     == gradient of the outer level function (function handle)
% grad_f     == gradient of the inner level function f (function handle)
% prox_g     == proximal operator of the function g (function handle)
% x0         == starting point 
% L_w        == Lipschitz constant of 'grad_f' 
% L_f        == Lipschitz constant of 'grad_w'
% sigma      == stron convexity constant of w
%
% varargin   == niter: number of iterations 
%               w    : outer level function (function handle) 
%                      to monitor the energy
%               F    : inner level function (function handle)
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
F       = @(x) pi;
w       = @(x) pi;
int     = 25;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Check, whether w is available
if (w(x0) ~= pi)
    fun_val_w = true;
else
    fun_val_w = false;
end

% Check, whether F is available
if (F(x0) ~= pi)
    fun_val_F = true;
else
    fun_val_F = false;
end

% Check whether history is requested
if (nargout - 1 == 1)
    if (fun_val_w && fun_val_F)
        hist = zeros(ceil(niter/int),2);
    else 
        error('Please provide the functions w and F.');
    end
end

% Initialize 
s = 1.99/(L_w + sigma); % Step size for outer problem
t = 0.99/L_f; % Step size for inner problem

x = x0;

for it = 1:niter 
    y = prox_g(x - t * grad_f(x));
    z = x - s * grad_w(x); 
    x = 1/it * z + (1-(1/it)) * y; 
    
    if (fun_val_w && mod(it,int) == 0)
        hist(it/int,1) = w(x);
        hist(it/int,2) = F(x);
        fprintf('It:%d. Outer level energy: %6.6d. Inner level energy: %6.6d.\n',it, hist(it/int,1), hist(it/int,2));
    end 
    
end

if (nargout - 1 == 1)
    varargout{1} = hist;
end

end

