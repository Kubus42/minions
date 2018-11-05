function [x, varargout] = il2_tgv_warm(f,op,lambda,L,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 50; 
int       = 1;
norm_type = 'iso';
alpha0    = 1;
alpha1    = sqrt(2);

% Step size 
tau   = 0.99/L;

% Compare to optimal 
load('CT_opt.mat');
en_opt = hist_opt.p(end);
n_opt = norm(ct_opt(:));

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

Atb = At(f);

x = Atb; x_old = x; x_hat = x;
z = grad(x); z_old = z; z_hat = z; 

% Warm starting 
y1_0 = grad(x); 
y2_0 = symgrad(z);

a = 2;
d = 1;
t_old = 1;
theta = 0;
it = 1;

px = sqrt(numel(x));
% Do the work 
while ( it <= niter )
    y     = x_hat - tau * At(A(x_hat)) + tau * Atb; 
    tol   = 1 / 2 / ((1 + 2 * theta)*(it+1)^2);
    y_tilde = cat(3,y,z_hat);
    x0 = prox_tv(y,0.02);
    [x,z,h,y1_0,y2_0] = prox_tgv(y_tilde,tau*lambda, 'norm_type',norm_type,'tol',tol,'y1_0',y1_0,'y2_0',y2_0,'x0',x0);
    t     = (((it+1) + a - 1)/a)^d;
    theta = (t_old - 1) / t;
    
    x_hat = x + theta * (x - x_old); 
    z_hat = z + theta * (z - z_old);
    x_old = x;
    z_old = z;
    
    if (isempty(hist) == 0 && mod(it,int) == 0)
        hist(it/int,1) = 0.5 * norm(vec(A(x)-f))^2 + lambda * tgv(x,z,'alpha0',alpha0,'alpha1',alpha1);
%         hist(it/int,2) = - 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
        inner_it(it/int,1) = h.iter;
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, hist(it/int,1));
        if exist('ct_opt')
            err(it/int,1) = norm(vec(x - ct_opt)) / n_opt;
            err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;   
        end

    end

    it    = it + 1;
end 

if (nargout - 1 == 1)
    hist_out.p        = reshape(nonzeros(hist),[numel(nonzeros(hist))/1,1]);
    hist_out.inner_it = nonzeros(inner_it);
    hist_out.errn     = nonzeros(err(:,1));
    hist_out.erren    = nonzeros(err(:,2));
    varargout{1}      = hist_out;
end





end
