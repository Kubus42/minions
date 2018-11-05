function [x, x_erg, varargout] = prox_icbtv_tv(f, q, alpha, w, varargin)
%DENOISING WITH A STRUCTURAL PRIOR
% 
% Solves min_{x} 0.5 ||x-f||^2 + alpha * ( w * TV(x) + (1-w) * ICBTV^q(x) ), 
%
% where TV and ICBTV can be anisotropic or isotropic (default).
%
% Input: 
% f        ==  data
% q        ==  subgradient vector field (s.t. -div(q) \in \partial TV(prior))
% alpha    ==  regularization parameter
% w        ==  weight between TV and ICBTV ( \in (0,1) )
% varargin ==  'norm_type' : 'aniso' / 'iso' 
%          ==  'niter'     :  maximal iterations 
%          ==  'tol'       :  tolerance for RMSE / primal-dual gap
%          ==  'int'       :  interval for RMSE and history
%          ==  'verbose'   :  true/false for output or not
%          ==  'scale_step':  scaling of the step sizes
%
% Output: 
% x         ==  last iterate after 'niter' iterations or stop criterion
% x_erg     ==  the ergodic averages of x 
% varargout ==  'hist'     : convergence history
%           ==  'z'        : the infimal convolution variable

% Defaults 
norm_type  = 'iso';
niter      = 5000; 
tol        = 1e-3;
int        = 50;
verbose    = true;
scale_step = 1;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Step sizes 
% (4.78 is an approximation of the norm of the operator 
% [grad, -grad; 0, grad; grad, 0] in 2 dimensions)
tau = 0.99/4.87 * scale_step;
sigma = tau/scale_step; 

% History?
if (nargout - 2 >= 1)
    hist = zeros(ceil(niter/int), 4);
else
    hist = [];
end

% Helper 
vec      = @(x) x(:);

% Initialize
x  = f; x_old = x; x_erg = 0; x_bar = x;
z  = f; z_old = z; z_bar = z;
y1 = grad(x); % grad(x-z)
y2 = y1; % grad(z)
y3 = y1; % grad(x)

rmse = 1e10;
px = numel(f);
it = 1;

% Do the work 
while ( it < niter && rmse > tol ) 
    
    % dual updates
    y1 = prox_dual_l1(y1 + sigma * grad(x_bar - z_bar) + alpha*(1-w)*q, alpha*(1-w), norm_type) - alpha*(1-w)*q;
    y2 = prox_dual_l1(y2 + sigma * grad(z_bar) - alpha*(1-w)*q, alpha*(1-w), norm_type) + alpha*(1-w)*q;
    y3 = prox_dual_l1(y3 + sigma * grad(x_bar), w*alpha, norm_type);
    
    % primal updates 
    x = (x + tau * (div(y1 + y3)) + tau * f) / (1 + tau);
    z = z + tau * div(y2 - y1);
    
    % ergodic
    x_erg = x_erg + x;
    
    % overrelaxation 
    x_bar = 2*x - x_old; 
    z_bar = 2*z - z_old;
    
    x_old = x; 
    z_old = z; 
    
    % Compute convergence criteria
    if ( mod(it,int) == 0 )
        pr = 0.5 * norm(vec(x-f), 2)^2 ... 
            + alpha*(1-w) * ( tv(x-z, norm_type) + tv(z, norm_type) ...
            - vec(q)' * vec(grad(x-z)) + vec(q)' * vec(grad(z)) ) ...
            + alpha*w * tv(x, norm_type);             
        dr = -0.5 * norm(vec(div(y1+y3)), 2)^2 - vec(div(y1+y3))' * vec(f) ...
            - vec(div(y2-y1))' * vec(z);
        constraint = norm(vec(div(y2 - y1))) / px;
        
        gap  = pr-dr; 
        rmse = sqrt(2*gap/px);
        
        % some output if wanted
        if verbose
            fprintf('It: %6.6d. RMSE.: %6.6d. Constraint: %6.6d.\n', it, rmse, constraint);
        end
        
        % history
        if ( isempty(hist) == 0 )  
            hist(it/int, 1) = rmse;
            hist(it/int, 2) = constraint;
            hist(it/int, 3) = pr;
            hist(it/int, 4) = dr;
        end
        
    end
    it = it + 1;
        
end

if (it == niter && rmse > tol)
    fprintf('RMSE did NOT converge below %8.2E. \n', tol);
end

if (nargout - 2 >= 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/4,4]);
    hist_out.rmse = hist(:, 1);
    hist_out.constraint = hist(: ,2);
    hist_out.pr = hist(: ,3);
    hist_out.dr = hist(: ,4);
    hist_out.iter = it;
    varargout{1} = hist_out;
    varargout{2} = z;
end

x_erg = x_erg/it;

end
