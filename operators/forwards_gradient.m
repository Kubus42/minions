function [DX,DY,DZ] = forwards_gradient(m,n,t)
%Gradient matrix with forwards differences

Dy=spdiags([-ones(m,1) ones(m,1)],[0 1],m,m);
Dy(m,:)=0;
Dx=spdiags([-ones(n,1) ones(n,1)],[0 1],n,n);
Dx(n,:)=0;
DX=kron(Dx,speye(m));
DY=kron(speye(n),Dy);

DX = kron(speye(t),DX);
DY = kron(speye(t),DY);

Dz=spdiags([-ones(t,1) ones(t,1)],[0 1],t,t);
Dz(t,:)=0;
DZ=kron(Dz,speye(m*n));
end

