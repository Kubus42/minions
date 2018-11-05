function [x,varargout] = ipd_l1_tv(f,op,lambda,varargin)

% Inexact primal-dual for the L1-TV model

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 1000; 
pdtol     = 1e-6; 
int       = 50; 
norm_type = 'iso';
C         = 1; % Constant for gap
alpha     = 1.01;
start     = 'warm';
mp        = false;

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
if (exist('tv_deblurring_l1_opt.mat', 'file') == 2)
    load('tv_deblurring_l1_opt.mat');
    en_opt = hist_opt.p(end);
    n_opt = norm(tvdbl1_opt(:));
end

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),3);
    err = zeros(ceil(niter/int),2);
    inner_it = zeros(ceil(niter/int),1);
else
    hist = [];
    err = [];
    inner_it = [];
end

% Define the operators
A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

% Initialize
x  = zeros(size(At(f))); x_old = x; 
y = A(x); y_old = y;
Ax = A(x); Ax_old = Ax; 
Ax_bar = Ax;
Aty = At(y); Aty_old = Aty;

% Overrelaxation
theta = 1; 
it = 1;
PDres = 1;

% Initialize warm start
if strcmp(start,'warm') == 1
    y0 = zeros(size(grad(x)));
    npx = numel(y0); % Number of pixels
end


tic;
% Do the work 
while ( it <= niter && PDres > pdtol )
   
    % Dual update
    y = prox_dual_l1(y + sigma * (Ax_bar - f), 1, 'aniso');    
    Aty = At(y);
    
    % Determine the constant for the gap (Villa/Salzo)
    if (it == 1)
        w = x - tau * Aty;       
        C =  lambda * tv(w);
    end
    
    % tau is to account for the multiplication with tau of the total energy
    tol = (C*tau)/it^alpha;
    
    % Primal update
    if (strcmp(start,'warm') ~=1 && ~ mp )
        [x,h]  = prox_tv(x - tau * Aty, tau * lambda,'tol',tol,'err','gap','tau',1/32); 
    else
        if mp
            [x,h,y0]  = prox_tv(x - tau * Aty, tau * lambda,'tol',1e-10*npx,'err','gap','y0',y0,'verbose',true,'niter',1);
        else
            [x,h,y0]  = prox_tv(x - tau * Aty, tau * lambda,'tol',tol,'err','gap','y0',y0);
        end
    end
    
    
    % Overrelaxation
    Ax = A(x);     
    Ax_bar = Ax + theta * (Ax - Ax_old); 
    
    
    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr  = norm(vec((x_old - x)/tau   - (Aty_old - Aty)),1)/numel(x);
        dr = norm(vec((y_old - y)/sigma -  (Ax_old - Ax)),1)/numel(y);
        PDres = pr + dr; 
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);       
        
        if (isempty(hist) == 0)
            p = norm(vec(A(x)-f),1) + lambda * tv(x,norm_type);
            hist(it/int,1) = p;
            hist(it/int,2) = 1; 
            hist(it/int,3) = PDres;
            inner_it(it/int,1) = h.iter;
            if ( exist('tvdbl1_opt','var') == 1 )
                err(it/int,1) = norm(vec(x - tvdbl1_opt)) / n_opt;
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

if (it == niter && PDres > pdtol)
    fprintf('PD residual did NOT converge below %8.2E. \n', pdtol);
end

if (nargout - 1 == 1)
    aux = reshape(nonzeros(hist),[numel(nonzeros(hist))/3,3]);
    hist_out.p = aux(:,1); 
    hist_out.d = aux(:,2); 
    hist_out.pdres = aux(:,3);
    hist_out.errn = err(:,1);
    hist_out.erren = err(:,2);
    hist_out.inner_it = nonzeros(inner_it);
    varargout{1} = hist_out;
end

toc;


end

