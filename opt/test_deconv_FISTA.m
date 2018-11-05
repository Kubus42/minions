close all;
clear;

% mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" ../rof/TV4_mex.cpp

addpath('../common');
addpath('../data');
addpath('../rof');

rng(1);
 
%% load image

g = double(imread('ju52.JPG'))/255.0;
%g = imresize(g, 1/8);
g = max(0,min(1,imresize(g, .2)));
[M, N, K] = size(g);

%% Load kernel
psf = mean(double(imread('kernel2.png'))/255.0, 3);
psf = psf/sum(psf(:));

%%
f = g;
f(:,:,1) = reshape(mex_convolution(reshape(g(:,:,1),M*N,1),psf,M,N), M,N);
f(:,:,2) = reshape(mex_convolution(reshape(g(:,:,2),M*N,1),psf,M,N), M,N);
f(:,:,3) = reshape(mex_convolution(reshape(g(:,:,3),M*N,1),psf,M,N), M,N);

f = f + randn(M,N,K)*0.01;
imshow(f);
%pause;
drawnow;

lambda = 2000;
tol = .01;

u = deblurMask(f, psf, 1/lambda, tol); %% sans masque ça ne
                                       %% marchera pas
                                       
imshow(u);
imwrite(u, '../results/ju52_deconv_FISTA.png');

%imwrite(f, '../results/ju52_deconv_input.png');
