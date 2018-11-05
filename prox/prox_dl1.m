function res = prox_dl1(f, v, alpha)
% Proximal operator for directional total variation.
%
% Solves 
%
% min_x 0.5 * |x - f|^2 + alpha * | || P_v^orth (x) || |_1, 
%
% where P_v^orth is the pointwise projection onto the orthogonal 
% complement of v.
%
% Input: 
% f          ==   data 
% v          ==   vector field for the projection
% 
% alpha      ==   regularization parameter

sz  = size(f);
dim = numel(sz); 

if (sz ~= size(v))
    error('Dimension mismatch between data and vector field.')
end

% Separate the data into two parts 
f1 = sum(real(f.*conj(v)),dim) .* v; % pointwise projection onto v 
f2 = f - f1; % the orthogonal part 

% Solve the shrinkage problem on the orthogonal part 
f2 = reshape(f2,[prod(sz(1:end-1)),sz(end)]);
norm_f2 = sqrt(sum(abs(f2).^2,2));
for i = 1:sz(end)
    f2(:,i) = max(0, 1-alpha./max(norm_f2,1e-9)) .* f2(:,i);
end
f2   = reshape(f2,sz);

% Put together
res = f1 + f2;


end
