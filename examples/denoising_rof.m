% some denoising with ROF

% load an image 
f = im2double(imread('Giraffe_bw.png'));

% add some noise 
sigma = 0.05; 
f_noisy = f + sigma * randn(size(f));

% show 
figure; 
subplot(121); 
imagesc(f); axis image; title('original');
subplot(122);
imagesc(shrinkImage(f_noisy,0,1)); axis image; title('noisy');
colormap gray; 

%% Do some reconstructions with different algorithms 
alpha = 0.3;

niter = 1000; 
int = 10; 

% nonsmooth primal-dual 
[u_pd, hist_u_pd] = rof(f, alpha, 'pd', 'niter', niter, 'int', int, 'verbose', true);

% nonsmooth fista (on the dual problem)
[u_fista, hist_u_fista] = rof(f, alpha, 'fista', 'niter', niter, 'int', int);

% (smooth) Huber-TV on the dual with fista
epsilon = 0.001;
[u_fista_smooth, hist_u_fista_smooth] = rof(f, alpha, 'fista', 'niter', niter, 'int', int, 'epsilon', epsilon, 'verbose', true);

% (smooth) Huber-TV with primal-dual
[u_pd_smooth, hist_u_pd_smooth] = rof(f, alpha, 'pd', 'niter', niter, 'int', int, 'epsilon', epsilon, 'verbose', true);

%% Show 
figure; 
subplot(241); 
imagesc(u_pd); axis image; colormap gray; title('pd');
subplot(242); 
imagesc(u_fista); axis image; colormap gray; title('fista');
subplot(243);
imagesc(u_fista_smooth); axis image; colormap gray; title('fista with Huber');
subplot(244);
imagesc(u_pd_smooth); axis image; colormap gray; title('pd with Huber');

% convergence
subplot(245);
plot(hist_u_pd.p); title('Energy');
subplot(246);
plot(hist_u_fista.p); title('Energy');
subplot(247);
plot(hist_u_fista_smooth.p); title('Energy');
subplot(248);
plot(hist_u_pd_smooth.p); title('Energy');













