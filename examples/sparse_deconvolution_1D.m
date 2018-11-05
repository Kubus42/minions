% Example for sparse deblurring/deconvolution in 1D

%% setup the operator and create some data
% create some sparse signal 
N = 200; 
f = zeros(N,1); 

% create some peaks
f(floor(N/3)) = 1;
f(floor(N/4)) = -0.5;
f(floor(2*N/3)) = 0.5; 

% create a Gaussian blurring operator 
hsize = 40; % kernel size 
sigma = 2.5; % standard deviation of the Gaussian

x = linspace(-hsize/2, hsize/2, hsize);
filt = exp(-x.^2 / (2 * sigma^2));
filt = filt / sum (filt); % normalize

op = @(x) conv(x, filt, 'same');
op_adj = @(x) conv(x, filt, 'same');

% apply to 'f' and add some noise
sigma_noise = 0.025; 
f_noisy = op(f) + sigma_noise * randn(size(f));

% show the original signal and the blurry/noisy data
figure; 
subplot(131); 
plot(f); axis([1,N,-0.75,1.2]); hold on;
plot(f_noisy); 
title('original + blurry/noisy');

%% do a reconstruction using FISTA
alpha = 0.025;
F = @(x) 0.5 * norm(op(x) - f_noisy,2)^2 + alpha*norm(x,1);
gradient = @(x) op_adj(op(x) - f_noisy); 
prox_op = @(x,tau) prox_l1(x, tau*alpha);

u = fista(gradient, prox_op, f_noisy, 1, 'F', F, 'int', 50); 

% show the result 
subplot(132); 
plot(u); axis([1,N,-0.75,1.2]); title('reconstruction');

%% do a debiasing step 
% compute the subgradient
p = -gradient(u) / alpha;

% correct numerical imperfections 
p = shrinkImage(p, -1, 1);

% do the debiasing 
gamma = 100; 
F_deb = @(x) 0.5 * norm(op(x) - f_noisy,2)^2 + gamma*(norm(x,1) - p'*x);
gradient_deb = @(x) op_adj(op(x) - f_noisy); 
prox_op_deb = @(x,tau) prox_l1(x + gamma*tau*p, tau*gamma);

u_deb = fista(gradient_deb, prox_op_deb, f, 1, 'F', F_deb, 'int', 100, 'niter', 10000); 

% show the result 
subplot(133); 
plot(u_deb); axis([1,N,-0.75,1.2]); title('debiased reconstruction');








