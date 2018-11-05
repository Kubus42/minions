function blur = createGaussianBlurringMatrix(sizeImg,hsize,sigma)
%
% Calculates a sparse matrix for blurring an image of size 'sizeImg' 
% with a Gaussian kernel of size 'hsize' and standard deviation
% 'sigma'. 
%
% Should only be used for the computation of step sizes 
% for preconditioning. For operators use 'createGaussianFilter' 
% and corresponding functions. 
%
% Input: 
% sizeImg == Size of the image to be blurred
% hsize   == Size of the Gaussian kernel
% sigma   == Standard deviation of the Gaussian  
%
% Output: 
% blur    == Sparse blurring matrix 

M = sizeImg(1); N = sizeImg(2); k = hsize; %Sinogram size and blur kernel size
h = fspecial('gaussian', k, sigma);
 
image = zeros(M,N);
blur = sparse(1,1,0,M*N,M*N, 2 * max(M,N)* N*M); 

fprintf('Calculate the Gaussian blur matrix.\n');
for i = 1:M
    fprintf(['Step ', num2str(i), ' of ', num2str(M), '.\n']);
    for j = 1:N
        image(i,j) = 1;
        aux = imfilter(image, h, 0, 'same');
        image(i,j) = 0;
        blur(:,(j-1)*M + i) = reshape(aux, [M*N 1]);
    end
end

