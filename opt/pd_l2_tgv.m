function [x,varargout] = pd_l2_tgv(f,op,lambda,varargin)

% Some description

 
% varargin   ==   'niter'    :  maximal iterations: 
%            ==   'tol'      :  tolerance for primal-dual residual
%            ==   'int'      :  interval for residual and history

% Defaults 
niter     = 1000; 
tol       = 1e-4; 
int       = 50; 
norm_type = 'iso';
alpha0    = 1;
alpha1    = sqrt(2);

% Step sizes

% Deblurring
% s = 1;
% tau   = s * 0.9/4.8622; 
% sigma = tau / s;

% Compare to optimal 
load('CT_opt.mat');
en_opt = hist_opt.p(end);
n_opt = norm(ct_opt(:));

% CT 
tau = 0.9 / 3.3705; 
sigma = tau;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec = @(x) x(:);

% History?
if (nargout - 2 == 1)
    hist = zeros(ceil(niter/int),3);
    err = zeros(ceil(niter/int),2);
else
    hist = [];
    err  = [];
end

% Initialize

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

B  = @(x) grad(x); 
Bt = @(x) -div(x);

C  = @(x) symgrad(x); 
Ct = @(x) -symdiv(x);

x  = At(f); x_old = x; 
z  = grad(x); z_old = z;
y1 = A(x); y1_old = y1;
y2 = B(x); y2_old = y2; 
y3 = C(z); y3_old = y3;

Ax = A(x); Ax_old = Ax; 
Bx = B(x); Bx_old = Bx;
Iz = z; Iz_old = Iz;
Cz = C(z); Cz_old = Cz;
 
Ax_bar = Ax;
Bx_bar = Bx;
Cz_bar = Cz;
Iz_bar = Iz;

Aty1 = At(y1); Aty1_old = Aty1;
Bty2 = Bt(y2); Bty2_old = Bty2;
Ity2 = y2; Ity2_old = Ity2;
Cty3 = Ct(y3); Cty3_old = Cty3;


theta = 1; 
it = 1;
PDres = 1;

% Do the work 
while ( it <= niter && PDres > tol )
   
    y1 = (y1 + sigma * (Ax_bar - f)) / (1 + sigma);
    y2 = prox_dual_l1(y2 + sigma * (Bx_bar - Iz_bar), lambda * alpha0, norm_type); 
    y3 = prox_dual_l1(y3 + sigma * Cz_bar, lambda * alpha1, norm_type); 
    
    Aty1 = At(y1);
    Bty2 = Bt(y2);
    Ity2 = y2;
    Cty3 = Ct(y3);
    
    x  = x - tau * (Aty1 + Bty2);
    z  = z - tau * (Cty3 - Ity2);
    
    Ax = A(x); 
    Bx = B(x); 
    Iz = z;
    Cz = C(z);
    
    Ax_bar = Ax + theta * (Ax - Ax_old); 
    Bx_bar = Bx + theta * (Bx - Bx_old);
    Iz_bar = Iz + theta * (Iz - Iz_old);
    Cz_bar = Cz + theta * (Cz - Cz_old);

    % Check primal-dual residual
    if ( mod(it,int) == 0 )
        pr1 = norm(vec((x_old - x)/tau   - (Aty1_old + Bty2_old - Aty1 - Bty2)),1)/numel(x);
        pr2 = norm(vec((z_old - z)/tau   - (-Ity2_old + Cty3_old - (-Ity2 + Cty3))),1)/numel(z);
        dr1 = norm(vec((y1_old - y1)/sigma -  (Ax_old - Ax)),1)/numel(y1);
        dr2 = norm(vec((y2_old - y2)/sigma -  (Bx_old - Iz_old - (Bx - Iz))),1)/numel(y2);
        dr3 = norm(vec((y3_old - y3)/sigma -  (Cz_old - Cz)),1)/numel(y3);
        pr = pr1 + pr2; 
        dr = dr1 + dr2 + dr3;
        PDres = pr + dr; 
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it, PDres);
        
        if (isempty(hist) == 0)
            hist(it/int,1) = 0.5 * norm(vec(A(x)-f))^2 + lambda * tgv(x,z,'alpha0',alpha0,'alpha1',alpha1);
            hist(it/int,2) = 1; %- 0.5 * norm(vec(div(y2)))^2 - sum(vec(div(y2).* f));
            hist(it/int,3) = PDres;
        end
        
            if exist('ct_opt')
            err(it/int,1) = norm(vec(x - ct_opt)) / n_opt;
            err(it/int,2) = abs(hist(it/int,1) - en_opt) / en_opt;   
        end
    
        
    end
    
    x_old = x; 
    z_old = z;
    
    y1_old = y1; 
    y2_old = y2;
    y3_old = y3;
    
    Ax_old = Ax; 
    Bx_old = Bx;
    Iz_old = Iz;
    Cz_old = Cz;
    
    Aty1_old = Aty1; 
    Bty2_old = Bty2;
    Ity2_old = Ity2;
    Cty3_old = Cty3;
    
    it    = it + 1;
end 

if (it == niter && PDres > tol)
    fprintf('PD residual did NOT converge below %8.2E. \n', tol);
end

switch nargout - 1
    case 1
        varargout{1} = z;
    case 2
        varargout{1} = z;
        hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/3,3]);
        hist_out.p     = hist(:,1);
        hist_out.d     = hist(:,2);
        hist_out.pdres = hist(:,3); 
        hist_out.errn  = nonzeros(err(:,1));
        hist_out.erren = nonzeros(err(:,2));
        varargout{2}   = hist_out;
end




end











