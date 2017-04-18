function [E_z, var_z] = transform_to_z(X,Y,n_components)

%this functions transform X,Y to latent variable z with
[A,B,r,U,V] = canoncorr(X,Y);

A = A(:,1:n_components);
B = B(:,1:n_components);

M1 = diag(sqrt(r));
M2 = M1;

M = vertcat(M1,M2);

Pd_square = diag(r)^2;

I_Pd2_inv = inv(eye(n_components) - Pd_square);
I_Pd2_inv_Pd = I_Pd2_inv*diag(r);

left = vertcat(I_Pd2_inv,I_Pd2_inv_Pd);
right = vertcat(I_Pd2_inv_Pd,I_Pd2_inv);

middle = horzcat(left,right);

mu1 = mean(X);
mu2 = mean(Y);

U1d = A'*(X-mu1)';
U2d = B'*(Y-mu2)';

U = vertcat(U1d,U2d);

E_z = (M'*middle*U)';
var_z = M'*middle*M;

end