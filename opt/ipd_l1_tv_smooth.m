function [x,x_avg,varargout] = ipd_l1_tv_smooth(f,op,lambda,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 1000; 
tol       = 1e-4; 
int       = 50; 
norm_type = 'iso';
epsilon   = 0.1;
alpha     = 1.1;
start     = 'warm';

% Deblurring
tau   = 0.99/1; 
sigma = tau;


% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);

% Compare to optimal 
if (exist('tv_deblurring_l1_opt_smooth.mat', 'file') == 2)
    load('tv_deblurring_l1_opt_smooth.mat');
    en_opt = hist_opt.p(end);
    n_opt = norm(tvdbl1_opt(:));
end

% History?
if (nargout - 2 == 1)
    hist = zeros(ceil(niter/int),4);
    err = zeros(ceil(niter/int),3);
    inner_it = zeros(ceil(niter/int),1);
else
    hist = [];
    err = [];
    inner_it = [];
end

% Initialize

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

x  = zeros(size(At(f))); x_old = x; 
x_avg = x;
y = A(x); y_old = y;
y_avg = y;

Ax = A(x); Ax_old = Ax; 
Ax_bar = Ax;

Aty = At(y); Aty_old = Aty;

sigma_avg = 0;

% Initialize warm start
if strcmp(start,'warm') == 1
    y0 = zeros(size(grad(x)));
end

it = 1;
PDres = 1;

% Do the work 
while ( it <= niter && PDres > tol )
   
    % Dual update
    y = prox_dual_l1(y + sigma * (Ax_bar - f), 1, 'aniso');    
    Aty = At(y);
    
    % Determine the constant for the gap (Villa/Salzo)
    if (it == 1)
        w = x - tau * Aty;       
        C =  lambda * tv(w) + epsilon/2 * norm(vec(w))^2;
    end
    
    tolerance = C*tau / it^(2 * alpha);
    
    % Primal update
    if (strcmp(start,'warm') == 1)
        [x,h,y0] = prox_tv_smooth(x - tau * Aty, tau * lambda,'tol',tolerance,'epsilon',epsilon*tau,'err','gap','x0',x,'y0',y0);
    else
        [x,h] = prox_tv_smooth(x - tau * Aty, tau * lambda,'tol',tolerance,'epsilon',epsilon*tau,'err','gap');
    end
    
    % Ergodic average 
    x_avg = x_avg + sigma * x;
    y_avg = y_avg + sigma * y;
    sigma_avg = sigma_avg + sigma;
    
    
    % Overrelaxation   
    theta = 1/sqrt(1 + epsilon * tau);
    tau   = tau * theta; 
    sigma = sigma / theta; 
    
    Ax = A(x);  
    Ax_bar = Ax + theta * (Ax - Ax_old); 

    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr  = norm(vec((x_old - x)/tau   - (Aty_old - Aty)),1)/numel(x);
        dr = norm(vec((y_old - y)/sigma -  (Ax_old - Ax)),1)/numel(y);
        PDres = pr + dr; 
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);
        
        
        
        if (isempty(hist) == 0)
            p = norm(vec(A(x)-f),1) + lambda * tv(x,norm_type) + epsilon/2 * norm(vec(x),2)^2;
            aux = x_avg / sigma_avg;
            p_avg = norm(vec(A(aux)-f),1) + lambda * tv(aux,norm_type) + epsilon/2 * norm(vec(aux),2)^2;

            hist(it/int,1) = p;
            hist(it/int,2) = 1; 
            hist(it/int,3) = PDres;
            hist(it/int,4) = p_avg;
            inner_it(it/int,1) = h.iter;
            if ( exist('tvdbl1_opt','var') == 1 )
                err(it/int,1) = norm(vec(x - tvdbl1_opt)) / n_opt;
                err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;
                err(it/int,3) = abs(hist(it/int,4) - en_opt) / en_opt;
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
    aux = reshape(nonzeros(hist),[numel(nonzeros(hist))/4,4]);
    hist_out.p = aux(:,1); 
    hist_out.d = aux(:,2); 
    hist_out.pdres = aux(:,3);
    hist_out.p_avg = aux(:,4);
    hist_out.errn = err(:,1);
    hist_out.erren = err(:,2);
    hist_out.erren_avg = err(:,3);
    hist_out.inner_it = nonzeros(inner_it);
    varargout{1} = hist_out;
end

x_avg = x_avg / sigma_avg;



end











