function rec = poisson_solver(gx,gy)

% pad size
wid = 1;
gx = padarray(gx,[wid wid]);
gy = padarray(gy,[wid wid]);

% define operators in spatial domain
nabla_x_kern = [0 0 0; -1 1 0; 0 0 0];
nabla_y_kern = [0 -1 0; 0 1 0; 0 0 0];

% specify boundary conditions
bc = 'symmetric';     % use DCT

% define adjoint operator
nablaT = @(gx,gy) imfilter(gx, rot90(nabla_x_kern,2), bc) + ...
                  imfilter(gy, rot90(nabla_y_kern,2), bc);

% genereate inverse kernel
[H,W] = size(gx);
[x_coord,y_coord] = meshgrid(0:W-1,0:H-1);
mat_x_hat = 2*cos(pi*x_coord/W) + 2*cos(pi*y_coord/H) - 4;
mat_x_hat(1) = 1;

% do inverse filtering
rec = idct2( dct2(nablaT(gx,gy)) ./ -mat_x_hat );

% get center
rec = rec(1+wid:end-wid,1+wid:end-wid);

% redeem on boundary
rec(1,:) = [];
rec(end,:) = [];
rec(:,1) = [];
rec(:,end) = [];
rec = padarray(rec, [1 1], 'replicate');
