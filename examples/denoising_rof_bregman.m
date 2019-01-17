% some denoising with Bregman-ROF

% load an image 
f = im2double(imread('cameraman.tif'));
sz = size(f);

% add some noise 
sigma = 0.05; 
f_noisy = f + sigma * randn(sz) * max(f(:));

% show 
figure; 
subplot(121); 
imagesc(f); axis image; title('original');
subplot(122);
imagesc(shrinkImage(f_noisy,0,1)); axis image; title('noisy');
colormap gray; 

%% 
alpha = 2;
niter = 1000; 
int = 100; 
niter_breg = 20;
epsilon = 0.01;

% subgradient
p1 = zeros(sz);
p2 = zeros(sz);

norm_type = 'iso';

figure; 
for i = 1:niter_breg
    disp(['Bregman iteration: ', num2str(i)]);
    [u, hist_u] = rof(f_noisy + alpha*p1, alpha, 'fista', 'niter', niter, 'int', int, 'verbose', false, 'norm_type', norm_type);
    [v, hist_v, y] = rof(f_noisy + alpha/i*p2, alpha/i, 'fista', 'niter', niter, 'int', int, 'verbose', false, 'norm_type', norm_type);
    p1 = p1 + (f_noisy-u)/alpha;
    p2 = artificial_subgradient(v,epsilon);
    subplot(231); imagesc(shrinkImage(u,0,1)); axis image; title('standard');
    subplot(234); imagesc(p1); axis image; title('standard');
    subplot(232); imagesc(shrinkImage(v,0,1)); axis image; title('crafted');
    subplot(235); imagesc(p2); axis image; title('crafted'); 
    subplot(233); imagesc(y(:,:,1)/alpha); axis image; title('x'); 
    subplot(236); imagesc(y(:,:,2)/alpha); axis image; title('y'); 
    colormap gray;   
    drawnow;
%     keyboard;
end


