function [x, varargout] = prox_dtv_pd(f, v, alpha, varargin)
%
% Solves min_x 0.5 |x-f|^2 + alpha * | ||P_v^orth(grad x)|| |_1, 
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic.
%
% Input: 
% f          ==   data
% v          ==   vector field for the projection
% alpha      ==   regularization parameter
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual

% Defaults 
niter     = 5000; 
tol       = 1e-5;
int       = 50;

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
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

% Initialize
x = f; x_old = x; 
y = grad(f);  

% Step sizes 
tau   = 1/sqrt(8);
sigma = 1/sqrt(8);
gamma = 1; 
theta = 1;

it = 1;
pdgap = 1;
% Do the work 
while ( it < niter && pdgap > tol )
    y_aux = y + sigma * grad(x + theta * (x - x_old));
    y = y_aux - sigma * prox_dl1(y_aux/sigma, v, alpha/sigma);
    x_old = x; 
    x = 1/(1+tau) * (x - tau * (-div(y)) + tau * f);
    
    % Check primal-dual gap
    if ( mod(it,int) == 0 )
        primal = 0.5 * norm(vec(x-f))^2 + alpha * dtv(x,v);
        dual   = - 0.5 * norm(vec(div(y)))^2 - sum(vec(div(y).* f));
        pdgap  = abs(primal - dual)/numel(x);

        if (isempty(hist) == 0)
            hist(it/int,1) = primal;
            hist(it/int,2) = dual;
        end
        fprintf('It: %6.6d. Duality gap.: %6.6d.\n',it,pdgap);
%         fprintf('PDgap.: %6.6d.\n',pdgap);
    end
    
    theta = 1/sqrt(1+2*gamma*tau);
    tau   = tau * theta;
    sigma = sigma / theta;

    it    = it + 1;
end 

if (it == niter && pdgap > tol)
    fprintf('PD gap did NOT converge below %8.2E. \n', tol);
end

if (nargout - 1 == 1)
    hist_out = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    varargout{1} = hist_out;
end

end
