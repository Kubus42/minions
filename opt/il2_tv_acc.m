function [x, varargout] = il2_tv_acc(b,op,lambda,L,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 50; 
int       = 1;
norm_type = 'iso';
eps       = 0; 

% Step size 
tau   = 0.99/L;

% Compare to optimal 
load('tv_deblurring_opt.mat');
en_opt = hist_opt.p(end);
n_opt = norm(tvdb_opt);

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),1);
    inner_it = zeros(ceil(niter/int),1);
    err = zeros(ceil(niter/int),2);
else
    hist = [];
    inner_it = [];
    err = [];
end

% Initialize

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

Atb = At(b);

x = Atb; x_old = x; x_hat = x;
t_old = 1;; 
it = 1;

% Do the work 
while ( it <= niter )
    % Gradient descent
    y     = x- tau * At(A(x)) + tau * Atb; 
    
    % Overrelaxation
    if ( it < niter) 
        t = (1 + sqrt(1 + 4 * t_old^2))/2; 
    else 
        t = (1 + sqrt(1 + 8 * t_old^2))/2;
    end
    
    theta1 = (t_old - 1) / t; 
    theta2 = t_old / t;
    theta3 = tau * (t_old - 1) / gamma / t;
    
    
    
    % Proximal step
    tol   = 1 / 2 /((1 + 2 * (theta1 + theta2))*(it+1)^2);
    [x,h] = prox_tv(y,tau*lambda,'norm_type',norm_type,'tol',tol);
    
    
    if (isempty(hist) == 0 && mod(it,int) == 0)
        hist(it/int,1) = 0.5 * norm(vec(A(x)-b))^2 + lambda * tv(x,norm_type);
%         hist(it/int,2) = - 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
        inner_it(it/int,1) = h.iter;
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, hist(it/int,1));
        if exist('tvdb_opt')
            err(it/int,1) = norm(x - tvdb_opt) / n_opt;
            err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;   
        end
    end

    it    = it + 1;
end 

if (nargout - 1 == 1)
    hist_out.p = reshape(nonzeros(hist),[numel(nonzeros(hist))/1,1]);
    hist_out.inner_it = nonzeros(inner_it);
    hist_out.errn = nonzeros(err(:,1));
    hist_out.erren = nonzeros(err(:,2));
    varargout{1}  = hist_out;
end



end
