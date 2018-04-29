function phi = tie_ours(I1, I0, I2, lambda, z)

% get dimensions
pdim = size(I0);

% get the DCT basis
H = pdim(1);
W = pdim(2);
[x_coord, y_coord] = meshgrid(0:W-1, 0:H-1);
mat_x_hat = 2*cos(pi*x_coord/W) + 2*cos(pi*y_coord/H) - 4;
mat_x_hat(1) = 1;       % avoid singularity; DC term

% define DCT Poisson solver (symmetric boundary condition)
% Teagues' method (double Poisson Equation with unit intensity):
% 
% Michael Reed Teague. "Deterministic phase retrieval: a Green’s function
% solution." JOSA 73.11 (1983): 1434-1441.
poisson = @(I0, g) idct2( dct2(I0 .* -g) ./ mat_x_hat ) / (2*pi/lambda);

% get g and I
g = (I2 - I1) / (2*z);

% solve for phi
phi = poisson(I0, g);
