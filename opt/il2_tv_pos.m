function [x, varargout] = il2_tv_pos(f,op,lambda,L,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 50; 
int       = 1;
norm_type = 'iso';

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

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),1);
else
    hist = [];
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

px = sqrt(numel(x));
% Do the work 
while ( it < niter )
    y     = x_hat - tau * At(A(x_hat)) + tau * Atf; 
    tol   = 1 / ((1 + 2 * theta)*(it+1)^2);
%     x     = TV4_mex(y,tau*lambda,eps,px * tol);
    x     = prox_tv_pos(y,tau*lambda,'norm_type',norm_type,'tol',tol);
    t     = (((it+1) + a - 1)/a)^d;
%     theta = (t_old - 1) / t;
    theta = 0;
    x_hat = x + theta * (x - x_old); 
    x_old = x;
    
    if (isempty(hist) == 0 && mod(it,int) == 0)
        hist(it/int,1) = 0.5 * norm(vec(A(x)-f))^2 + lambda * tv(x,norm_type);
%         hist(it/int,2) = - 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
    end

    it    = it + 1;
end 

if (nargout - 1 == 1)
    hist_out = reshape(nonzeros(hist),[numel(nonzeros(hist))/1,1]);
    varargout{1} = hist_out;
end




end
