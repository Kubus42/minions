function [x, varargout] = rof(f, alpha, alg, varargin)
%
% Solves min_{x \in C} 0.5 ||x-f||_2^2 + alpha * | ||grad x|| |_1,
%
% where ||.|| can be an l1-norm or l2-norm, i.e.
% anisotropic (default) or isotropic. The minimization is done either
%   - via an accelerated forward-backward splitting (FISTA) on the DUAL
%     problem. Here the initialization 'y0' has to be of the size of
%     grad(f). If you want to use Huber-ROF, choose epsilon > 0.
%   - via accelerated primal-dual. If you want to use Huber-ROF, 
%     choose epsilon > 0.
%
% Input:
% f          ==   data
% alpha      ==   regularization parameter
% alg        ==   'fista' or 'pd'
%
% for 'fista'
% varargin   ==   'y0'       :  initialization (dual variable)
%            ==   'norm_type': 'aniso' / 'iso'
%            ==   'niter'    :  maximal iterations
%            ==   'err'      :  error criterion, either 'rmse' or 'gap'
%            ==   'tol'      :  tolerance for 'err'
%            ==   'int'      :  interval for 'err' and history
%            ==   'P'        :  function handle: projection onto C
%            ==   'epsilon'  :  smoothing parameter for Huber-ROF
%            ==   'verbose'  :  output (yes == 'true' / no == 'false')
%
% for 'pd'
% varargin   ==   'x0'       :  initialization (primal variable)
%            ==   'y0'       :  initialization (dual variable)
%            ==   'norm_type': 'aniso' / 'iso'
%            ==   'niter'    :  maximal iterations
%            ==   'err'      :  error criterion, either 'rmse' or 'gap'
%            ==   'tol'      :  tolerance for 'err'
%            ==   'int'      :  interval for 'err' and history
%            ==   'P'        :  function handle: projection onto C
%            ==   'epsilon'  :  smoothing parameter for Huber-ROF
%            ==   'verbose'  :  output (yes == 'true' / no == 'false')

% Some parameters
px       = numel(f);
gap      = 1e10;
stop_tol = 1e10;

% Helper
vec = @(x) x(:);

% Defaults
norm_type = 'aniso';
niter     = 10000;
tol       = 1e-3;
int       = 25;
P         = @(x) x;
err       = 'rmse';
verbose   = false;
epsilon   = 0;

% History?
if (nargout == 2 || nargout == 3)
    hist = zeros(ceil(niter/int),2);
else
    hist = [];
end

switch alg
    case 'fista'
        % Defaults
        y0        = [];
       
        % Overload
        if isempty(varargin) == 0
            for i=1:2:length(varargin) % overwrites default parameter
                eval([varargin{i},'=varargin{',int2str(i+1),'};']);
            end
        end
        
        % Initialize
        if (isempty(y0) == 1)
            y = zeros(size(grad(f)));
        else
            y = y0;
        end
        
        % step size
        tau = 0.99/(4*numel(size(f)));
        
        y_old = y; y_hat = y;
        div_y = div(y);
        div_y_old = div_y;
        div_y_hat = div_y;
        
        % Some parameters
        mu = epsilon/alpha; % Strong convexity
        q  = (tau*mu) / (1 + tau*mu); % Related parameter
        % Starting overrelaxation step size
        if q > 0
            t_old = 1/sqrt(q);
        else
            t_old = 1;
        end
        
        it = 1;
        % Do the work
        while ( it < niter && stop_tol > tol )
            
            z = y_hat + tau * grad( P(f + div_y_hat )); % Gradient descent
            y = prox_dual_l1(z/(1+mu*tau),alpha,norm_type); % Proximal step
            
            t = (1-q*t_old^2 + sqrt((1-q*t_old^2)^2 + 4 * t_old^2)) / 2; % Overrelaxation step size 1
            theta = ((t_old-1)/t) * (1 + (1-t)*tau*mu); % Overrelaxation step size 2
            t_old = t;
            
            % Overrelaxation
            y_hat = y + theta * (y - y_old);
            y_old = y;
            div_y = div(y);
            div_y_hat = div_y + theta * (div_y - div_y_old);
            div_y_old = div_y;
            
            if ( mod(it,int) == 0 )
                x = P(f + div_y);
                if epsilon > 0
                    p = 0.5 * norm(vec(x-f),2)^2 + alpha * tv(x,norm_type,epsilon);
                    d = 0.5 * norm(vec( P(f + div_y ) - (f + div_y) ),2)^2 ...
                        - 0.5 * norm(vec( f + div_y ),2)^2 + 0.5 * norm(vec(f),2)^2 ...
                        - mu/2 * norm(vec(y),2)^2;
                    gap = abs(p-d);
                else
                    p = 0.5 * norm(vec(x-f),2)^2 + alpha * tv(x,norm_type);
                    d = 0.5 * norm(vec( P(f + div_y ) - (f + div_y) ),2)^2 ...
                        - 0.5 * norm(vec( f + div_y ),2)^2 + 0.5 * norm(vec(f),2)^2;
                    gap = abs(p-d);
                end
                
                if (verbose && strcmp('rmse',err) == 1)
                    fprintf('It: %6.6d. RMSE.: %6.6d.\n',it,sqrt(2*gap/px));
                elseif (verbose && strcmp('gap',err) == 1)
                    fprintf('It: %6.6d. Duality gap.: %6.6d.\n',it,gap);
                end
                
                if ( isempty(hist) == 0 )
                    hist(it/int,1) = p;
                    hist(it/int,2) = d;
                end
                
                if (strcmp('rmse',err) == 1)
                    stop_tol = sqrt(2*gap/px);
                elseif (strcmp('gap',err) == 1)
                    stop_tol = gap;
                end
                
            end
            it = it + 1;
            
        end
        
        if (it == niter && stop_tol > tol)
            fprintf('Did NOT converge below %8.2E. \n', tol);
        end
        
        % Output
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
        
        % Solution
        x = P(f + div(y));
        
    case 'pd'
        % Defaults   
        x0        = [];
        y0        = [];
        
        % Overload
        if isempty(varargin) == 0
            for i=1:2:length(varargin) % overwrites default parameter
                eval([varargin{i},'=varargin{',int2str(i+1),'};']);
            end
        end
        
        % Initialize
        % primal
        if (isempty(x0) == 1)
            x = f; x_old = x;
        else
            x = x0; x_old = x;
        end
        % dual
        if (isempty(y0) == 1)
            y = zeros(size(grad(f)));
        else
            y = y0;
        end
        
        % Step sizes
        if ( epsilon == 0 )
            tau   = 0.99/sqrt(4*numel(size(f)));
            sigma = 0.99/sqrt(4*numel(size(f)));
            gamma = 1; % strong convexity
            theta = 1;
        else
            L     = sqrt(4*numel(size(f)));
            gamma = 1; 
            mu    = 2*sqrt(gamma*epsilon/alpha) / L;
            tau   = mu / (2 * gamma);
            sigma = mu / (2 * epsilon/alpha);
            theta = 1 / (1 + mu);
        end
        
        
        it = 1;
        gap = 1;
        % Do the work
        while ( it < niter && stop_tol > tol )
            y = prox_dual_l1((y + sigma * grad(x + theta * (x - x_old))) / (1+sigma*epsilon/alpha), alpha, norm_type);
            x_old = x;
            x = P(1/(1+tau) * (x - tau * (-div(y)) + tau * f));
            
            % Check primal-dual gap
            if ( mod(it,int) == 0 )
                if ( epsilon > 0 )
                    p   = 0.5 * norm(vec(x-f))^2 + alpha * tv(x,norm_type,epsilon);
                    d   = - 0.5 * norm(vec(div(y)))^2 - sum(vec(div(y).* f)) - epsilon / (2*alpha) * norm(vec(y))^2;
                    gap = abs(p-d);
                else
                    p   = 0.5 * norm(vec(x-f))^2 + alpha * tv(x,norm_type);
                    d   = - 0.5 * norm(vec(div(y)))^2 - sum(vec(div(y).* f));
                    gap = abs(p-d);
                end
                
                if (verbose && strcmp('rmse',err) == 1)
                    fprintf('It: %6.6d. RMSE.: %6.6d.\n',it,sqrt(2*gap/px));
                elseif (verbose && strcmp('gap',err) == 1)
                    fprintf('It: %6.6d. Duality gap.: %6.6d.\n',it,gap);
                end
                
                if ( isempty(hist) == 0 )
                    hist(it/int,1) = p;
                    hist(it/int,2) = d;
                end
                
                if (strcmp('rmse',err) == 1)
                    stop_tol = sqrt(2*gap/px);
                elseif (strcmp('gap',err) == 1)
                    stop_tol = gap;
                end
                
                
            end
            
            % Acceleration
            if ( epsilon == 0 )
                theta = 1/sqrt(1+2*gamma*tau);
                tau   = tau * theta;
                sigma = sigma / theta;
            end
            
            it    = it + 1;
        end
        
        if (it == niter && stop_tol > tol)
            fprintf('Did NOT converge below %8.2E. \n', tol);
        end
        
        % Output
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
            varargout{2}  = struct('primal',x,'dual',y);
        end
        
end
