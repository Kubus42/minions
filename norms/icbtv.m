function [val, varargout] = icbtv(x, p, varargin)
%
% Solves 
%
% min_z [ D_TV^p(x-z,v) + D_TV^{-p}(z,-v) ] = min_z ICB_TV^p(x,v)
%
% where D_TV^p(x,v) = TV(x) - <p,v> and p \in \partial TV(v).
%
% 
% Input: 
%
% x          == input image
% p          == subgradient of v
% 
% varargin   == niter: maximal number of iterations 
%               tol  : tolerance for PD residual   
%               norm_type : anistropic or isotropic tv

% Defaults 
niter     = 1e5;
tol       = 1e-5;
norm_type = 'aniso';
int       = 50;

% Adaptivity
gamma = 0.5;
delta = 1.5;
eta   = 0.95;
sc    = 1; 

dim = ndims(x);
% Fix dimension if 1-dimensional
if (dim == 2 && ndims(x') == 1)
    x = x';
    p = p';
end

% History?
if (nargout - 1 == 2)
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

% Step sizes 
tau = 0.1; 
sigma = 0.1;
% tau   = 1/sqrt(2 * dim);
% sigma = 1/sqrt(2 * dim);

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Do the job
vec = @(x) x(:);

% Precompute the gradient of x 
grad_x = grad(x);

% Initialize 
z  = x; z_old = z; 
y1 = zeros(size(grad(z))); y1_old = y1;  
y2 = y1; y2_old = y2;

it = 1;
PDres = 1;
% Do the work 
while ( it < niter && PDres > tol )
    
    grad_z = grad(2*z - z_old);
    y1 = prox_dual_l1(y1 + sigma * (grad_z - grad_x), 1, norm_type); 
    y2 = prox_dual_l1(y2 + sigma * grad_z, 1, norm_type); 
    z_old = z;
    z  = z + tau * (div(y1+y2) - 2 * p);
    
    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr = norm(vec((z_old - z)/tau - div(-y1_old-y2_old) + div(-y1-y2)),1);
        dr1 = norm(vec((y1_old - y1)/sigma - grad(z_old - z)),1);
        dr2 = norm(vec((y2_old - y2)/sigma - grad(z_old - z)),1);
        dr = dr1 + dr2; 
        PDres = pr/numel(z) + dr/(numel(y1) + numel(y2)); 
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);
        
        % Adaptive Stepsize
        if (pr > sc * delta * dr)
            tau   = tau / (1 - gamma); % Increase primal stepsize
            sigma = sigma * (1 - gamma); % Decrease dual stepsize
            gamma = eta * gamma; % Decrease adaptivity
        elseif (pr < sc / delta * dr)
            tau   = tau * (1 - gamma); % Decrease primal stepsize
            sigma = sigma / (1 - gamma); % Increase dual stepsize
            gamma = eta * gamma; % Decrease adaptivity
        end
        
        % History
        if (isempty(hist) == 0)
            hist(it/int,1) = tv(x-z,norm_type) + tv(z,norm_type) - trace(p' * (x-2*z));
            hist(it/int,2) = PDres;
        end
    
        
    end
    y1_old = y1; 
    y2_old = y2; 

    it     = it + 1;
end 

if (it == niter && PDres > tol)
    fprintf('PD residual did NOT converge below %8.2E. \n', tol);
end

narg = nargout - 1;

switch (narg)
    case 1
        varargout{1} = z;
    case 2 
        varargout{1} = z;
        hist_out = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
        varargout{2} = hist_out;
end

% Attention: Not clear for 3 dimensions
val = tv(x-z) + tv(z) - trace(p' * (x-2*z));







end

