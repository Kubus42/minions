function [S, varargout] = symgrad(I)
%SYMMETRIZED GRADIENT
% 
% Attention: Only for two dimensional images and 
% I = [ I(:,:,1), I(:,:,2) ]

I1_grad = grad(I(:,:,1));
I2_grad = grad(I(:,:,2));

mixed = 0.5 * (I1_grad(:,:,2) + I2_grad(:,:,1));

S = cat(3, I1_grad(:,:,1), mixed, mixed , I2_grad(:,:,2));

if (nargout - 1 == 1) 
    varargout{1} = [I1_grad(:,:,1), mixed; mixed , I2_grad(:,:,2)];
end
    
end