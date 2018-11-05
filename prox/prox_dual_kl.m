function x = prox_dual_kl(y, f, alpha, sigma)
%
% Proximal operator for the dual of the Kullback-Leibler divergence, i.e.,  
% the convex conjugate of 
%
% alpha * ( sum y - f log(y) ).
%
% Input: 
% y          ==   input of the proximal operator 
% f          ==   "data" used in the Kullbac-Leibler divergence 
% alpha      ==   regularization parameter (in front of KL)
% sigma      ==   step size parameter of the proximal operator

x = (alpha + y)/2 - sqrt(1/4 * (y - alpha).^2 + sigma * alpha * f); % Kullback-Leibler
    
end
