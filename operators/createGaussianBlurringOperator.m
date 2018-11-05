function [Op, Op_adj] = createGaussianBlurringOperator(sizeImg,hsize,sigma)

% Creates an operator and its adjoint for a blurring with a 
% Gaussian kernel of size 'hsize' and standard deviation 'sigma'. 
%
% Input: 
% sizeImg == Image size
% hsize   == Size of the Gaussian kernel
% sigma   == Standard deviation of the Gaussian
%
% Output: 
% Op      == Blurrding operator 
% Op_adj  == Its adjoint operator

kernel = fspecial('gaussian', hsize, sigma);

fftFilter = zeros(sizeImg);
fftFilter(1:size(kernel,1),1:size(kernel,2)) = kernel;

% Center the kernel
fftFilter = circshift(fftFilter,-(size(kernel,1)-1)/2,1);
fftFilter = circshift(fftFilter,-(size(kernel,2)-1)/2,2);

% Precalculate FFT
fftFilter = fftn(fftFilter);
fftFilterC = conj(fftFilter);

% Setup the operators
Op = @(x) ifftn(fftn(reshape(x,sizeImg)) .* fftFilter);
Op_adj = @(y) ifftn(fftn(reshape(y,sizeImg)) .* fftFilterC);

end
