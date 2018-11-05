function [x, varargout] = prox_tv_smooth(f, alpha, varargin)
%
% Solves min_x 0.5 |x-f|^2 + alpha * | ||grad x|| |_1 + 0.5 * epsilon * |x|^2, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Input: 
% f          ==   data
% alpha      ==   regularization parameter
% varargin   ==   'norm_type': 'aniso' / 'iso' 
%            ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual

% Defaults 
niter     = 5000; 
norm_type = 'aniso';
tol       = 1e-3;
int       = 1;
epsilon   = 0.1;
x0        = [];
y0        = [];
err       = 'gap';
verbose   = false;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);

% History?
if (nargout >= 2)
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

% Initialize
if (isempty(y0) == 1)
    x = zeros(size(f));
    x_old = x;
    y = zeros(size(grad(f))); 
else
    x = x0;
    x_old = x0;
    y = y0;
end

% Step sizes 
tau   = 1/sqrt(8);
sigma = 1/sqrt(8);
gamma = 0.99 * (1 + epsilon); 
theta = 1;

px = numel(f);
stop_tol = 1e10;
it = 1;
gap = 100;
% Do the work 
while ( it <= niter && stop_tol > tol )
    y = prox_dual_l1(y + sigma * grad(x + theta * (x - x_old)), alpha, norm_type); 
    x_old = x; 
    x = (x - tau * (-div(y)) + tau * f) / (1 + tau + tau * epsilon);
    
    % Check primal-dual gap
    if ( mod(it,int) == 0 )
        div_y = div(y);
        primal = 0.5 * norm(vec(x-f))^2 + alpha * tv(x,norm_type) + epsilon/2 * norm(vec(x),2)^2;
        dual   = 1/(1+epsilon) * vec(-div_y)'*vec(f) - 1/(1+epsilon) * norm(vec(-div_y))^2 + ... 
            1/(2 * (1+epsilon)^2) * norm(vec(div_y - epsilon * f))^2 + 0.5 * epsilon * norm(vec(f + div_y)/(1+epsilon))^2;
        gap  = abs(primal - dual)/numel(x);

        if (verbose && strcmp('rmse',err) == 1)
            fprintf('It: %6.6d. RMSE.: %6.6d.\n',it,sqrt(2*gap/px));
        elseif (verbose && strcmp('gap',err) == 1)
            fprintf('It: %6.6d. Duality gap.: %6.6d.\n',it,gap);
        end
        
        if (isempty(hist) == 0)
            hist(it/int,1) = primal;
            hist(it/int,2) = dual;
        end
        
        if (strcmp('rmse',err) == 1)
            stop_tol = sqrt(2*gap/px);
        elseif (strcmp('gap',err) == 1)
            stop_tol = gap;
        end
    end
    
    theta = 1/(1+2*gamma*tau);
    tau   = tau * theta;
    sigma = sigma / theta;

    it    = it + 1;
end 

if (it == niter && gap > tol)
    fprintf('PD gap did NOT converge below %8.2E. \n', tol);
end

if (nargout - 1 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.d    = hist(:,2);
    hist_out.iter = it-1;
    hist_out.gap  = gap;
    hist_out.rmse = sqrt(2*gap/px);
    varargout{1}  = hist_out;
end

if (nargout - 2 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.d    = hist(:,2);
    hist_out.iter = it-1;
    hist_out.gap  = gap;
    hist_out.rmse = sqrt(2*gap/px);
    varargout{1}  = hist_out;
    varargout{2}  = y;
end

end