function [x, varargout] = l2_l1(f, op, alpha, varargin)
%
% Solves min_x 0.5 ||Ax-f||_2^2 + alpha * || x ||_1, 
%
% where ||.|| can be anisotropic (default) or isotropic. 
%
% Input: 
% f          ==   data
% alpha      ==   regularization parameter
% varargin   ==   'norm_type': 'aniso' / 'iso' 
%            ==   'niter'    :  maximal iterations 
%            ==   'tol'      :  tolerance for RMSE
%            ==   'int'      :  interval for RMSE and history

% Defaults 
norm_type = 'aniso';
niter     = 500; 
tol       = 1e-6;
int       = 1;
tau       = 0.99;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper
vec      = @(x) x(:);

% History?
if (nargout == 2)
    hist = zeros(ceil(niter/int),1);
else
    hist = [];
end

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

% Initialize 
x = At(f); x_old = x; x_hat = x;

% Some parameters
stop_tol = 1e10;
t_old = 1;
it = 1;
p_old = 1e10;

% Do the work 
while ( it <= niter )% && stop_tol > tol )
    
    % Gradient descent and proximal step
    z = x_hat - tau * At( A(x_hat) - f);
    x = prox_l1(z,tau*alpha,norm_type);
    
    % Overrelaxation
    t = (1 + sqrt(1 + 4*t_old^2))/2;
    theta = (t_old-1)/t; % Overrelaxation step size 2
    t_old = t;
    
    x_hat = x + theta * (x-x_old);
    x_old = x;
    
    if ( mod(it,int) == 0 )
        
        p = 0.5 * norm(vec(A(x)-f),2)^2 + alpha * sum(vec(abs(x)));
        fprintf('It: %6.6d. Energy.: %6.6d.\n',it,p);
               
        if ( isempty(hist) == 0 )        
            hist(it/int,1) = p;
        end

        stop_tol = abs(p - p_old);
        p_old = p;
    end
    it = it + 1;
        
end

if (nargout - 1 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.iter = it-1;
    varargout{1}  = hist_out;
end


end
