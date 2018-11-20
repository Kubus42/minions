% Some example code for denoising with a prior, either with directional TV
% or the infimal convolution of two TV Bregman distances 
clear; close all; clc; 

% Load some images 
load('pet_gt.mat');
load('mri_gt.mat');
uc = groundTruthPET;
vc = groundTruthMRI;

% load a red/blue colormap
load('blue_red_colormap.mat');

norm_type = 'iso';

%% Create a noisy version of u
sigma = 0.2;
un = uc + sigma * randn(size(uc)) * max(uc(:));
figure; subplot(131); 
imagesc(un); colormap gray; axis image; title('noisy');

%% Create a gradient of the MR image 
eta = 0.05;
grad_vc = gradient_direction(vc,eta);

% compute a subgradient 
beta = 0.01;
[vc_rof, ~, q] = rof(vc, beta, 'fista', 'niter', 20000, 'verbose', 1, 'tol', 1e-5, 'norm_type', norm_type, 'epsilon', 0.01);
subgrad_vc = q/beta;

%% Do the denoising with dTV
alpha = 5;
niter = 2000;
[u, ~] = prox_dtv_pd(un,grad_vc,alpha,'niter',niter);

% Compare to a standard ROF reconstruction
u2 = rof(un,2,'pd','niter',niter,'verbose',true);

% Plot
subplot(132); 
imagesc(shrinkImage(abs(u),0,10)); colormap gray; axis image; title('with prior');
subplot(133); 
imagesc(shrinkImage(u2,0,10)); colormap gray; axis image; title('rof');
figure;
imagesc(diffImage(abs(u), uc, [0, 10])); colormap(cmap); axis image; title('difference "with prior" and "ground truth"');

%% do it with ICBTV
alpha = 5;
niter = 5000;
w = 0.1;
[u, u_erg, hist_u, z] = prox_icbtv_tv(un, subgrad_vc, alpha, w, 'niter', niter, 'int', 100, 'norm_type', norm_type);

figure; 
subplot(131);
imagesc(shrinkImage(u, 0, 10)); axis image; colormap gray; title('w/ prior: iterate');
subplot(132); 
imagesc(shrinkImage(u_erg, 0, 10)); axis image; colormap gray; title('w/ prior: ergodic');
subplot(133); plot(hist_u.rmse); title('evolution RMSE');

figure; 
imagesc(diffImage(u, uc, [0, 10]) + 0.5); colormap(cmap); axis image; title('w/ prior: diff. to ground truth'); 

%% Try a reconstruction with a shifted version of the prior 
s = [2, 1]; % in px
vc_shift = imtranslate(vc, s); 

% Compute a subgradient
grad_vc_shift = gradient_direction(vc_shift, eta);
[vc_rof_shift, ~, q_shift] = rof(vc_shift, beta, 'fista', 'verbose', 1, 'tol', 1e-5, 'norm_type', norm_type, 'epsilon', 0.001);
subgrad_vc_shift = q_shift/beta;

% show
figure; 
subplot(131); 
imagesc(subgrad_vc_shift(:,:,1)); axis image; title('x gradient');
subplot(132); 
imagesc(subgrad_vc_shift(:,:,2)); axis image; title('y gradient');
subplot(133); 
imagesc(sqrt(sum(subgrad_vc_shift.^2,3))); axis image; title('norm gradient');

%% reconstruct
alpha = 10;
w = 0.1;
[u_shift, hist_u_shift] = prox_icbtv_tv(un, subgrad_vc_shift, alpha, w, 'niter', niter, 'int', 100); 

% show 
figure; 
imagesc(shrinkImage(u_shift, 0, 10)); colormap gray; axis image; title('with prior');

figure;
imagesc(diffImage(u_shift, uc, [0, 10]) + 0.5); colormap(cmap); axis image; title('diff. to ground truth'); 

%% Try a rotation 
gamma = 2; % angle for rotation
vc_rot = imrotate(vc, gamma, 'bilinear', 'crop');

eta = 0.05;
grad_vc_rot = gradient_direction(vc_rot, eta);
[vc_rof_rot, ~, q_rot] = rof(vc_rot, beta, 'fista', 'niter', 20000, 'verbose', 1, 'tol', 1e-6, 'norm_type', norm_type, 'epsilon', 0.001);
subgrad_vc_rot = q_rot/beta;

niter = 5000;
[u_rot, hist_u_rot] = prox_icbtv_tv(un, subgrad_vc_rot, alpha, w, 'niter', niter); 

figure; 
imagesc(shrinkImage(abs(u_rot), 0, 10)); colormap gray; axis image; title('with prior');

figure; 
imagesc(diffImage(abs(u_rot), uc, [0, 10]) + 0.5); colormap(cmap); axis image; title('diff. to ground truth');


