function G = grad(I)
%DIVERGENCE OPERATOR
% 
% Computes the gradient of 'I' in 1D, 2D or 3D. 
% 'grad' is adjoint to '-div' (mind the sign!)
%

nd=ndims(I);
ndt=nd;
for s=1:nd
    if size(I,s)==1
        ndt=ndt-1;
    end
end
nd=ndt;

%Gradient
switch nd
  case 1
    n  = max(size(I));
    dx = [2:n 1];
    G  = I(dx)-I;
    G(end) = 0;
  case 2
    [n,m] = size(I);
    dx = [2:n 1];
    dy = [2:m 1];
    G = cat(nd+1, I(dx,:)-I, I(:,dy)-I);
    G(end, :, 1) = 0;
    G(:, end, 2) = 0;
  case 3
    [n,m,t] = size(I);
    dx = [2:n 1];
    dy = [2:m 1];
    dz = [2:t 1];
    G = cat(nd+1, I(dx,:,:)-I, I(:,dy,:)-I, I(:,:,dz)-I);
    G(end, :, :, 1) = 0;
    G(:, end, :, 2) = 0;
    G(:, :, end, 3) = 0;
  otherwise
    error(['Grad for dimension ' num2str(nd) ' not implemented']);
end

end

