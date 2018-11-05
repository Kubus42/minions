function [S, varargout] = symdiv(I)
%SYMMETRICAL DIVERGENCE 
%
% Attention: Only for two dimensional images and 
% I = [ I(:,:,1), I(:,:,2), I(:,:,3), I(:,:,4) ]

I1_divx = divx(I(:,:,1));
I2_divx = divx(I(:,:,2));
I2_divy = divy(I(:,:,2));
I3_divx = divx(I(:,:,3));
I3_divy = divy(I(:,:,3));
I4_divy = divy(I(:,:,4));

S1 = I1_divx + 0.5 * I2_divy + 0.5 * I3_divy; 
S2 = I4_divy + 0.5 * I2_divx + 0.5 * I3_divx;

S = cat(3,S1,S2);

if (nargout - 1 == 1) 
    varargout{1} = [S1,S2];
end
    
end