function [x,z, varargout] = ikl_icbtv(f,op,lambda,L,w,p,q,varargin)

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

% Compare to optimal 
% load('pet_tv_opt.mat');
% en_opt = hist_opt(10000,1);
% n_opt = norm(u_opt(:));

% History?
if (nargout - 2 == 1)
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

x = ones(size(Atf));  x_old = x; x_hat = x;
z = zeros(size(Atf)); z_old = z; z_hat = z;
a = 2;
d = 1;
t_old = 1;
theta = 0;
it = 1;

dv = zeros([size(grad(x)),3]);

% Do the work 
while ( it <= niter )
    Ax_hat = A(max(x_hat,1e-6));
%     Ax_hat = A(x_hat);
    y1      = x_hat - tau .* At(1 - (f./ Ax_hat).*(Ax_hat>0)); 
    tol   = 1 / 2 / ((1 + 2 * theta)*(it+1)^2);
    y     = cat(3,y1,z_hat);
%     tol = 1e-3;
    [x,z,h,dv] = prox_icbtv2(y,lambda * tau,w,p,q,'init',dv,'norm_type',norm_type,'niter',150);
%     [x,z,h,dv] = prox_icbtv2(y,lambda * tau,w,p,q,'init',dv,'norm_type',norm_type,'tol',tol);
%     [x,z,h] = prox_icbtv(y,lambda * tau,w,p,q,'norm_type',norm_type,'tol',tol);
   
%     figure; subplot(131); plot(h.rmse); title('rmse');
%     subplot(132); plot(h.p); title('p');
%     subplot(133); plot(h.p-h.d); title('gap');
%     close;
    
    t     = (((it+1) + a - 1)/a)^d;
    theta = (t_old - 1) / t;         
    t_old = t;
    
    x_hat = x + theta * (x - x_old); 
    z_hat = z + theta * (z - z_old);
    x_old = x;
    z_old = z;

    if (isempty(hist) == 0 && mod(it,int) == 0)
        Ax = A(x);
        hist(it/int,1) = sum(Ax) - sum(f) ... 
                - sum(f(f~=0).*log(Ax(f~=0))) ...
                + sum(f(f~=0).*log(f(f~=0))) ... 
                + lambda * (- vec(w*p + (1-w)*q)' * vec(x) ... 
                + (2-2*w) * vec(q)' * vec(z) ...
                + w * tv(x,norm_type) + (1-w) * tv(x-z,norm_type) ... 
                + (1-w) * tv(z,norm_type) );
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, hist(it/int,1));
        
        inner_it(it/int,1) = h.iter;
        
        if exist('u_opt')
            err(it/int,1) = norm(vec(x - u_opt)) / n_opt;
            err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;   
        end
    end
    it    = it + 1;
end 

if (nargout - 2 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/1,1]);
    hist_out.p = hist;
    hist_out.inner_it = nonzeros(inner_it);
    if (isempty(err) == 0)
        hist_out.errn = nonzeros(err(:,1));
        hist_out.erren = nonzeros(err(:,2));
    end
    varargout{1}  = hist_out;
end




end
