function [x, varargout] = ikl_tv(f,op,lambda,L,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 50; 
int       = 1;
norm_type = 'iso';

% Compare to optimal 
load('pet_tv_opt.mat');
en_opt = hist_opt(10000,1);
n_opt = norm(u_opt(:));

% Step size 
tau   = 0.99/L;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);
P   = @(x) max(x,1e-6);

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

Atf = At(f);

x = Atf; x_old = x; x_hat = x;
a = 2;
d = 1;
t_old = 1;
theta = 0;
it = 1;
dual = grad(x);

% Do the work 
while ( it <= niter )
    Ax_hat = A(x_hat);
    y      = x_hat - tau * At(1 - (f./ Ax_hat).*(Ax_hat>0)); 
    tol    = 1 / 2 /((1 + 2 * theta)*(it+1)^2);
    [x,h]      = prox_tv(y,tau*lambda,'norm_type',norm_type,'tol',tol,'P',P);
    t      = (((it+1) + a - 1)/a)^d;
    theta  = (t_old - 1) / t;
    x_hat = x + theta * (x - x_old); 
    x_old = x;
    
    if (isempty(hist) == 0 && mod(it,int) == 0)
        Ax = A(x);
        hist(it/int,1) = sum(Ax) - sum(f) - sum(f(f~=0).*log(Ax(f~=0))) ...
                + sum(f(f~=0).*log(f(f~=0))) + lambda * tv(x,norm_type);
%         hist(it/int,2) = - 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, hist(it/int,1));
        inner_it(it/int,1) = h.iter;
        
        if exist('u_opt')
            err(it/int,1) = norm(vec(x - u_opt)) / n_opt;
            err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;   
        end
        
    end

    t_old = t;
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
