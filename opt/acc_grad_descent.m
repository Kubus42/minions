function x = acc_grad_descent(grad_f, x0, L, varargin) 
%
% Solves 
%
% min_x f(x) 
%
% for f differentiable with L-Lipschitz gradient using accelerated 
% gradient descent.
%
% Input 
% 
% grad_f     == gradient of the function to be minimized (function handle)
% x0         == starting point 
% L          == Lipschitz constant of 'grad_f' 
%
% varargin   == niter: number of iterations

% Defaults 
niter     = 500; 

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Initialize 
t_old = 0; 
x_old = x0; 
x     = x0;

tau = 0.99/L;

for i = 1:niter 
    t = (1 + sqrt(1 + 4 * t_old^2)) / 2;
    y = x + (t_old - 1)/t * (x - x_old); 
    x = y - tau * grad_f(y);
    
    t_old = t; 
    x_old = x;
    
end

fprintf('||grad_f(x)|| = %6.6d.\n', norm(grad_f(x)));



end

