function [x,varargout] = pd_l1_tv_smooth(f,op,lambda,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 1000; 
tol       = 1e-4; 
int       = 50; 
epsilon   = 0.1; 
norm_type = 'iso';
L         = 2.815; 


% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Deblurring
tau   = 0.99/L; 
% tau = 0.5;
sigma = (1-tau*epsilon)/tau/L^2;

% Helper 
vec = @(x) x(:);

% Compare to optimal 
if (exist('tv_deblurring_l1_opt_smooth.mat', 'file') == 2)
    load('tv_deblurring_l1_opt_smooth.mat');
    en_opt = hist_opt.p(end);
    n_opt = norm(tvdbl1_opt(:));
end

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),3);
    err = zeros(ceil(niter/int),2);
else
    hist = [];
    err = [];
end

% Initialize

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

B  = @(x) grad(x); 
Bt = @(x) -div(x);


x  = zeros(size(At(f))); x_old = x; 
y1 = A(x); y1_old = y1;
y2 = B(x); y2_old = y2; 

Ax = A(x); Ax_old = Ax; 
Bx = B(x); Bx_old = Bx;
Ax_bar = Ax;
Bx_bar = Bx;

Aty1 = At(y1); Aty1_old = Aty1;
Bty2 = Bt(y2); Bty2_old = Bty2;

theta = 1; 

it = 1;
PDres = 1;
p = 100;
% Do the work 
while ( it <= niter && PDres > tol )
   
    y1 = prox_dual_l1(y1 + sigma * (Ax_bar - f), 1, 'aniso');
    y2 = prox_dual_l1(y2 + sigma * Bx_bar, lambda, norm_type); 
    
    Aty1 = At(y1);
    Bty2 = Bt(y2);
    
    x  = (1-tau*epsilon) * x - tau * (Aty1 + Bty2);
    
    Ax = A(x); 
    Bx = B(x); 
    
    % Overrelaxation   
    theta = 1/sqrt(1 + epsilon * tau);
    tau   = tau * theta; 
    sigma = sigma / theta;
    
    Ax_bar = Ax + theta * (Ax - Ax_old); 
    Bx_bar = Bx + theta * (Bx - Bx_old);
    
    
    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr  = norm(vec((x_old - x)/tau   - (Aty1_old + Bty2_old - Aty1 - Bty2)),1)/numel(x);
        dr1 = norm(vec((y1_old - y1)/sigma -  (Ax_old - Ax)),1)/numel(y1);
        dr2 = norm(vec((y2_old - y2)/sigma -  (Bx_old - Bx)),1)/numel(y2);
        dr = dr1 + dr2;
        PDres = pr + dr; 
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);
        
        % Adaptive Stepsize
%         if (pr > sc * delta * dr)
%             tau   = tau / (1 - gamma); % Increase primal stepsize
%             sigma = sigma * (1 - gamma); % Decrease dual stepsize
%             gamma = eta * gamma; % Decrease adaptivity
%         elseif (pr < sc / delta * dr)
%             tau   = tau * (1 - gamma); % Decrease primal stepsize
%             sigma = sigma / (1 - gamma); % Increase dual stepsize
%             gamma = eta * gamma; % Decrease adaptivity
%         end

        
        
        if (isempty(hist) == 0)
            p = norm(vec(A(x)-f),1) + lambda * tv(x,norm_type) + epsilon/2 * norm(vec(x),2)^2;
            hist(it/int,1) = p;
            hist(it/int,2) = 1; 
            hist(it/int,3) = PDres;
            if ( exist('tvdbl1_opt','var') == 1 )
                err(it/int,1) = norm(vec(x - tvdbl1_opt)) / n_opt;
                err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;
            end
        end
    
        
    end
    
    x_old = x; 
    
    y1_old = y1; 
    y2_old = y2;
    
    Ax_old = Ax; 
    Bx_old = Bx; 
    
    Aty1_old = Aty1; 
    Bty2_old = Bty2;
    it    = it + 1;
end 

if (it == niter && PDres > tol)
    fprintf('PD residual did NOT converge below %8.2E. \n', tol);
end

if (nargout - 1 == 1)
    aux = reshape(nonzeros(hist),[numel(nonzeros(hist))/3,3]);
    hist_out.p = aux(:,1); 
    hist_out.d = aux(:,2); 
    hist_out.pdres = aux(:,3);
    hist_out.errn = err(:,1);
    hist_out.erren = err(:,2);
    varargout{1} = hist_out;
end




end










