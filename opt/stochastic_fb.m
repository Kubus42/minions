%% Test forward backward
clear; close all; clc; 
load('CTop.mat'); 
nC = normest(C); 
C = C/nC; % normalize

u0 = phantom(256);
sz = size(u0);
f0 = C * u0(:); 
f0 = f0 + 0.05 * randn(size(f0)) * mean(f0); 
%% Determine some blocks

nblocks = 23; 

% Every n-th line 
szC = size(C); 

blocks = zeros(szC(1)/nblocks,nblocks);
for b = 1:nblocks
    blocks(:,b) = b:nblocks:szC(1);
end



%% Some initialization
niter = 500;
z = zeros(prod(sz),nblocks); % Store gradients
Ctf = C' * f0;

x = zeros(prod(sz),1);
tau = 1/nblocks;
alpha = 0.001;
y0 = zeros([sz,2]);


%% Determine different step size
nC = zeros(nblocks,1);
for b = 1:nblocks
    nC(b) = normest(C(blocks(:,b),:));
end

tau = 1/nblocks/max(nC);

%% Do the job

for it = 1:niter   
    
    block = mod(it,nblocks)+1;    
    % Update z
    z(:,block) = C(blocks(:,block),:)' * (C(blocks(:,block),:) * x);
       
    % Do the gradient descent
    y = x - tau * sum(z,2) + tau * Ctf;
     
    % Do the prox 
    [x,~,y0] = prox_tv(reshape(y,sz),tau * alpha,'niter',10,'y0',y0);
    x = x(:);
    
    imagesc(reshape(x,sz)); colormap gray; axis image; drawnow; 
    
    disp(it);
    
end




