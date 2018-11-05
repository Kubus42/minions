function [x,varargout] = spd_kl_tv(f,A,blur,lambda,sz,S,varargin)

% Solves the minimization problem
%
% \min_{u>=0} KL(f,Au) + lambda * TV(u)
%
% with a stochastic primal-dual method. The cell array S determins the sampling.    
%
% Inputs: 
%
% f          == PET data
% A          == PET operator matrix
% lambda     == regularization parameter
% sz         == size of the image 'u'
% S          == cell array with sampling subsets
% 
% varargin   ==   'niter'    :  maximal iterations (epochs)
%            ==   'tol'      :  tolerance for primal-dual gap
%            ==   'int'      :  interval for residual and history
%            ==   'norm_type': 'iso' or 'aniso'

% Defaults 
niter     = 100; 
tol       = 1e-4; 
int       = 1; 
norm_type = 'iso';

% Overload
if isempty(varargin) == 0  
    for i=1:2:length(varargin) % overwrites default parameter
	eval([varargin{i},'=varargin{',int2str(i+1),'};']);
    end
end

% Determine the step sizes
gamma = 0.99; 
szS   = size(S); 
norms = cell(szS);
sigma = cell(szS);
N     = S{1,2};
for i = 1:szS(1)
    norms{i,1} = normest(A(S{i,1},:) * blur); 
    sigma{i,1} = gamma / norms{i,1};
end

tau = gamma / (N * max(cell2mat(norms)));

% Helper 
vec = @(x) x(:);
prox_kl = @(x,f,sigma) 0.5 * (x + 1 - sqrt( (x - 1).^2 + 4 * sigma .* f));  
P = @(x) max(x,1e-6);

% Compare to optimal 
load('pet_tv_opt.mat');
en_opt = hist_opt(10000,1);
n_opt  = norm(u_opt(:));

% History?
if (nargout - 1 == 1)
    hist = zeros(ceil(niter/int),2);
    err  = zeros(ceil(niter/int),2);
else
    hist = [];
    err  = [];
end

% Initialize

x = ones(size(blur' * (A'* f)));  
y = zeros(size(f)); y_old = y;

Aty = zeros(size(x));
Aty_bar = Aty;

theta = 1; 
it    = 1;
gap   = 1e10;

y0 = [];
% 
% fig1 = figure; 
% fig2 = figure; 

% Do the work 
while ( it <= niter && gap > tol )
    
    for s = 1:szS(1)
        % Primal update        
        [x,~,y0] = prox_tv(reshape(x - tau * Aty_bar,sz),tau * lambda,'y0',y0,'niter',10,'P',P); 
%         figure(fig1); imagesc(x); caxis([0,10]); colormap gray; drawnow; 
        x = vec(x);
        
        % Dual update
        aux = y(S{s,1}) + sigma{s,1} * A(S{s,1},:) * (blur * x); 
        y(S{s,1}) = prox_kl(aux,f(S{s,1}),sigma{s,1});
        
%         figure(fig2); imagesc(reshape(y,[344,252])); drawnow; pause;
%         Aty_bar = blur' * (A' * y); % No relaxation
        
        % Overrelaxation
        aux = blur' * ( A(S{s,1},:)' * ( y(S{s,1}) - y_old(S{s,1})) );
        Aty_bar = Aty + (1 + N*theta) * aux;        
        Aty = aux + Aty;
        y_old = y;
    end

    % Compute stats
    if (isempty(hist) == 0)
        Ax    = A * (blur * x);
        p     = sum(Ax) - sum(f) - sum(f(f~=0).*log(Ax(f~=0))) ...
            + sum(f(f~=0).*log(f(f~=0))) + lambda * tv(reshape(x,sz),norm_type);   
%         d_aux = 1 - y;        
%         d_aux = f(f~=0) ./ d_aux(f~=0);
%         d     = - sum(f(f~=0) .* (log(d_aux)-1));
%         gap   = abs(p-d) / numel(x); 
        fprintf('It: %6.6d. Energy: %6.6d.\n',it, p);
        hist(it/int,1) = p;
        hist(it/int,2) = 1; 
        err(it/int,1)  = norm(x - u_opt(:)) / n_opt;
        err(it/int,2)  = abs(hist(it/int,1) - en_opt) / en_opt;
    end

    it    = it + 1;
end 

if (it > niter && gap > tol)
    fprintf('PD gap did NOT converge below %8.2E. \n', tol);
end

if (nargout - 1 == 1)
    aux = reshape(nonzeros(hist),[numel(nonzeros(hist))/2,2]);
    hist_out.p = aux(:,1); 
    hist_out.d = aux(:,2); 
    hist_out.errn = err(:,1);
    hist_out.erren = err(:,2);
    varargout{1} = hist_out;
end

x = reshape(x,sz);


end











