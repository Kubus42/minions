function [x, x_avg ,varargout] = ipd_l2_tv_acc_warm(f,op,lambda,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 1000; 
tol       = 1e-4; 
int       = 50; 
norm_type = 'iso';
q         = 1;

% Deblurring
tau   = 0.99/1; 
sigma = tau;

gamma = 1;


% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);

% Compare to optimal 
if (exist('tv_deblurring_opt.mat', 'file') == 2)
    load('tv_deblurring_opt.mat');
    en_opt = hist_opt.p(end);
    n_opt = norm(tvdb_opt(:));
end

% History?
if (nargout - 2 == 1)
    hist = zeros(ceil(niter/int),3);
    err = zeros(ceil(niter/int),2);
    inner_it = zeros(ceil(niter/int),1);
else
    hist = [];
    err = [];
    inner_it = [];
end

% Initialize

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

x  = zeros(size(At(f))); x_old = x; x_avg = x;
y = A(x); y_old = y; y_avg = y;

Ax = A(x); Ax_old = Ax; 
Ax_bar = Ax;

Aty = At(y); Aty_old = Aty;


it = 1;
PDres = 1;
px = numel(f);

tau_avg = tau; 

y0 = grad(x);

% Do the work 
while ( it <= niter )
   
    % Dual update
    y = (y + sigma * (Ax_bar - f)) / (1 + sigma);    
    Aty = At(y);
    
    if (it == 1)
        % Determine the magnitude of the gap
        C = tau*lambda * tv(x-tau*Aty,norm_type);        
    end
    tol = C/it^q;
    
    % Primal update
    [x,h,y0]  = prox_tv(x-tau*Aty,tau*lambda,'tol',tol,'y0',y0,'err','gap');
    
    % Ergodic average 
    x_avg = x_avg + tau * x;
    y_avg = y_avg + tau * y;
    tau_avg = tau_avg + tau;
    
    % Update parameters
    theta = 1 / sqrt(1 + gamma * sigma); 
    sigma = theta * sigma; 
    tau = tau / theta;
    
    % Overrelaxation
    Ax = A(x);     
    Ax_bar = Ax + theta * (Ax - Ax_old); 
    
    
    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr  = norm(vec((x_old - x)/tau   - (Aty_old - Aty)),1)/numel(x);
        dr = norm(vec((y_old - y)/sigma -  (Ax_old - Ax)),1)/numel(y);
        PDres = pr + dr; 
%         fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);
        
        
        
        if (isempty(hist) == 0)
            p = 0.5 * norm(vec(A(x)-f),2)^2 + lambda * tv(x,norm_type);
            fprintf('It: %6.6d. Energy: %6.6d.\n',it, p);
            hist(it/int,1) = p;
            hist(it/int,2) = 1; 
            hist(it/int,3) = PDres;
            inner_it(it/int,1) = h.iter;
            if ( exist('tvdb_opt','var') == 1 )
                err(it/int,1) = norm(vec(x - tvdb_opt)) / n_opt;
                err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;
            end
        end
    
        
    end
    
    x_old = x; 
    y_old = y; 
    Ax_old = Ax;    
    Aty_old = Aty; 
    it    = it + 1;
end 

if (it == niter && PDres > tol)
    fprintf('PD residual did NOT converge below %8.2E. \n', tol);
end

if (nargout - 2 == 1)
    aux = reshape(nonzeros(hist),[numel(nonzeros(hist))/3,3]);
    hist_out.p = aux(:,1); 
    hist_out.d = aux(:,2); 
    hist_out.pdres = aux(:,3);
    hist_out.errn = err(:,1);
    hist_out.erren = err(:,2);
    hist_out.inner_it = nonzeros(inner_it);
    varargout{1} = hist_out;
end

x_avg = x_avg / tau_avg;



end











