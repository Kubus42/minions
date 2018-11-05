function [x, varargout] = prox_tv_prec(f, alpha, varargin)
%
% Solves min_{x} 0.5 ||x-f||_K^2 + alpha * | ||grad x|| |_1, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic, and K is a preconditioning matrix 
% with positive entries (default = identity).
%
% Input: 
% f          ==   data
% alpha      ==   regularization parameter
% varargin   ==   'norm_type': 'aniso' / 'iso' 
%            ==   'niter'    :  maximal iterations 
%            ==   'tol'      :  tolerance for RMSE
%            ==   'int'      :  interval for RMSE and history
%            ==   'K'        :  preconditioning matrix  

% Defaults 
norm_type = 'iso';
niter     = 5000; 
tol       = 1e-3;
int       = 1;
K         = 1;
y0        = [];

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec      = @(x) x(:);
normK    = @(x) sqrt( sum (vec(x.* (K.* x)) ) );
P = @(x) x;

% Step sizes 
iK = 1./K;
nn  = max(vec(iK));
tau = 1/(8*nn);
% tau = 1/(16*nn);

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

% Initialize
if (isempty(y0) == 1)
    y = grad(f); 
else
    y = y0;
end
    
y_old = y; y_hat = y;
div_y = div(y); 
div_y_old = div_y;
div_y_hat = div_y;
t_old = 1;
gap = 1e10;
px = numel(f);

it = 1;
% Do the work 
while ( it < niter && sqrt(2*gap/px ) > tol )
    z = y_hat + tau * grad( P(f + iK .* div_y_hat )); 
    y = prox_dual_l1(z,alpha,norm_type);    
    t = (1 + sqrt(1 + 4 * t_old^2)) / 2;
    
    y_hat = y + (t_old-1)/t * (y - y_old);
    y_old = y;
    
    div_y = div(y);
    div_y_hat = div_y + (t_old-1)/t * div_y - (t_old-1)/t * div_y_old;
    div_y_old = div_y;
    
    t_old = t;

    if ( mod(it,int) == 0 )
        x = P(f+ iK.*div_y);
        p = 0.5 * normK(x-f)^2 + alpha * tv(x,norm_type);
        d = 0.5 * normK( P(f+iK.*div_y ) - (f + iK.*div_y) )^2 ...
            - 0.5 * normK( f + iK.* div_y )^2 + 0.5 * normK(f)^2;
        gap = abs(p-d);
        fprintf('It: %6.6d. RMSE.: %6.6d.\n',it,sqrt(2*gap/px));
        if ( isempty(hist) == 0 )        
            hist(it/int,1) = p;
            hist(it/int,2) = d;
        end
        
    end
    it = it + 1;
        
end

if (it == niter && sqrt(2*gap/px) > tol)
    fprintf('RMSE did NOT converge below %8.2E. \n', tol);
end

if (nargout - 1 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.d    = hist(:,2);
    hist_out.iter = it;
    hist_out.rmse = sqrt(2*gap/px);
    varargout{1}  = hist_out;
end

if (nargout - 2 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.d    = hist(:,2);
    hist_out.iter = it;
    hist_out.rmse = sqrt(2*gap/px);
    varargout{1}  = hist_out;
    varargout{2}  = y;
end


x = P(f + iK .* div(y));

end