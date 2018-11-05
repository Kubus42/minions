function [x, varargout] = il2_tv(b,op,lambda,L,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 50; 
int       = 1;
norm_type = 'iso';
eps       = 0; 
alpha     = 0.99;
start     = 'warm';

% Step size 
tau   = 0.99/L;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Compare to optimal 
if (exist('tv_deblurring_l1_opt.mat', 'file') == 2)
    load('tv_deblurring_opt.mat');
    en_opt = hist_opt.p(end);
    n_opt = norm(tvdb_opt(:));
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
a = 2.1;
d = 1;
t_old = 1;
theta = 0;
it = 1;

% Initialize warm start
if strcmp(start,'warm') == 1
    y0 = zeros(size(grad(x)));
    npx = numel(y0); % Number of pixels
end

% Do the work 
while ( it <= niter )
    y     = x_hat - tau * At(A(x_hat)) + tau * Atb;     
        % Determine the constant for the gap (Villa/Salzo)
    if (it == 1)       
        C =  lambda * tv(y);
    end
    
    tol   = (C*tau)/(it^(1 + 2 * alpha) * log(it)^4); % tau is to account for the multiplication with tau of the total energy
    if (strcmp(start,'warm') ~=1)
        [x,h] = prox_tv(y,tau*lambda,'norm_type',norm_type,'tol',tol,'err','gap','tau',1/32);
    else
        [x,h,y0] = prox_tv(y,tau*lambda,'norm_type',norm_type,'tol',tol,'err','gap','y0',y0);
%         [x,h,y0] = prox_tv(y,tau*lambda,'norm_type',norm_type,'tol',1e-10,'err','gap','y0',y0,'niter',100);
    end
    t     = (((it+1) + a - 1)/a)^d;
    theta = (t_old - 1) / t;
    t_old = t;
    x_hat = x + theta * (x - x_old); 
    x_old = x;
    
    if (isempty(hist) == 0 && mod(it,int) == 0)
        hist(it/int,1) = 0.5 * norm(vec(A(x)-b))^2 + lambda * tv(x,norm_type);
%         hist(it/int,2) = - 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
        inner_it(it/int,1) = h.iter;
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, hist(it/int,1));
        if exist('tvdb_opt')
            err(it/int,1) = norm(vec(x - tvdb_opt)) / n_opt;
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
