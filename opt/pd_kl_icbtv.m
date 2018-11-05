function [x, z, varargout] = pd_kl_icbtv(f,op,gamma,w,p,q, varargin)
%
% CHANGE DESCRIPTION
%
% 
% Solves min_{x>=0, z} 0.5 ||x-f||_K^2 + alpha * | ||grad x|| |_1, 
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
niter     = 1000; 
tol       = 1e-3;
int       = 10;

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Helper 
vec      = @(x) x(:);
P = @(x) max(x,1e-10);

A  = @(x) op{1,1}(x);
At = @(x) op{1,2}(x);

B  = @(x) grad(x); 
Bt = @(x) -div(x); 

% Step sizes 
tau = 0.99/4.8632;
sigma = tau; 
theta = 1;

% History?
if (nargout - 2 == 1)
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

% Initialize
x  = At(f); x_old = x; 
z  = zeros(size(x)); z_old = z;

y1 = A(x);   y1_old = y1;
y2 = B(x);   y2_old = y2; 
y3 = B(x-z); y3_old = y3; 
y4 = B(z);   y4_old = y4;

Ax = A(x); Ax_old = Ax;
Bx = B(x); Bx_old = Bx;
Bz = B(z); Bz_old = Bz; 

Ax_bar = Ax;
Bx_bar = Bx; 
Bz_bar = Bz;

Aty1 = At(y1); Aty1_old = Aty1;
Bty2 = Bt(y2); Bty2_old = Bty2;
Bty3 = Bt(y3); Bty3_old = Bty3;
Bty4 = Bt(y4); Bty4_old = Bty4;

% gap = 1e10;

it = 1;
% Do the work 
while ( it < niter)
    
    y1 = prox_dual_kl(y1 + sigma * Ax_bar,f,1/gamma,sigma);
    y2 = prox_dual_l1(y2 + sigma * Bx_bar, w, norm_type); 
    y3 = prox_dual_l1(y3 + sigma * (Bx_bar-Bz_bar), 1-w, norm_type); 
    y4 = prox_dual_l1(y4 + sigma * (Bz_bar), 1-w, norm_type); 
    
    Aty1 = At(y1);
    Bty2 = Bt(y2); 
    Bty3 = Bt(y3); 
    Bty4 = Bt(y4); 
    
    x = P( x - tau * (Aty1 + Bty2 + Bty3 - (w*p + (1-w)*q)) );
    z = z - tau * (-Bty3 + Bty4 + 2*(1-w)*q);
    
    Ax = A(x);
    Bx = B(x); 
    Bz = B(z); 
    
    Ax_bar = Ax + theta * (Ax - Ax_old);
    Bx_bar = Bx + theta * (Bx - Bx_old);
    Bz_bar = Bz + theta * (Bz - Bz_old); 
    

    if ( mod(it,int) == 0 )
        pr =  sum(Ax) - sum(f) ... 
            - sum(f(f~=0).*log(Ax(f~=0))) ...
            + sum(f(f~=0).*log(f(f~=0))) ...
            + gamma * (-vec(w*p + (1-w)*q)' * vec(x) ... 
            + (2-2*w) * vec(q)' * vec(z) ...
            + w * tv(x,norm_type) + (1-w) * tv(x-z,norm_type) ... 
            + (1-w) * tv(z,norm_type) );
        dr = 1e-10;
        
        pres1 = norm(vec((x_old - x)/tau - (Aty1_old + Bty2_old + Bty3_old) + (Aty1 + Bty2 + Bty3)),1);
        pres2 = norm(vec((z_old - z)/tau - (-Bty3_old + Bty4_old) + (-Bty3 + Bty4)),1);
        
        % DUAL RESIDUAL
        dr1 = norm(vec((y1_old - y1)/sigma - Ax_old + Ax),1);
        dr2 = norm(vec((y2_old - y2)/sigma - Bx_old + Bx),1);
        dr3 = norm(vec((y3_old - y3)/sigma - (Bx_old - Bz_old) + (Bx - Bz)),1);
        dr4 = norm(vec((y4_old - y4)/sigma - Bz_old + Bz),1);
        
        nPrimalPixels = numel(x) + numel(z);
        nDualPixels = numel(y1) + numel(y2) + numel(y3) + numel(y4);
        
        PRes = (pres1 + pres2)/nPrimalPixels;
        DRes = (dr1 + dr2 + dr3 + dr4)/nDualPixels;
        
        PDres = PRes + DRes;  

        
        
        fprintf('It: %6.6d. PDres.: %6.6d.\n',it,PDres);
        if ( isempty(hist) == 0 )        
            hist(it/int,1) = pr;
            hist(it/int,2) = dr;
        end
        
        
    end

    x_old = x; 
    z_old = z; 
    
    y1_old = y1; 
    y2_old = y2; 
    y3_old = y3;
    y4_old = y4;
    
    Ax_old = Ax;
    Bx_old = Bx; 
    Bz_old = Bz;
    
    Aty1_old = Aty1;
    Bty2_old = Bty2;
    Bty3_old = Bty3; 
    Bty4_old = Bty4;
    
    it = it + 1;
        
end

% if (it == niter && sqrt(2*gap/px) > tol)
%     fprintf('RMSE did NOT converge below %8.2E. \n', tol);
% end

if (nargout - 2 == 1)
    hist = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p    = hist(:,1);
    hist_out.d    = hist(:,2);
    hist_out.iter = it;
    hist_out.rmse = 1; % sqrt(2*gap/px);
    varargout{1}  = hist_out;
end

end