function phi = main_wavefront_solver(img, beta, opt, warping_iter)
% [in] img      the stacked reference and captured image pair
% [in] beta     smoothness paramter \beta
% [in] opt      solver options:
%      opt.mu   [vector] proximal paramter at each pyramid level
%      opt.L    [cell] additional boundary unknowns at each pyramid level 
%               (better larger than [2 2] for each cell element)
%      opt.isverbose    [bool] is the solver verbose or not
%      opt.ls   [char] which linear solver to use; 'ADMM' or 'CG'
%      opt.iter [double] linear solver iterations
%      warping_iter [vector] warping iterations at each pyramid level
% [out] phi     the recovered wavefront

% set default values
if ~isfield(opt,'mu')
    opt.mu = 100;
end
if ~isfield(opt,'L')
    opt.L = repmat({[2 2]}, [1 numel(opt.mu)]);
end
if ~isfield(opt,'isverbose')
    opt.ls = 0;
end
if ~isfield(opt,'ls') || ((opt.ls ~= 'ADMM') && (opt.ls ~= 'CG'))
    opt.ls = 'ADMM';
    opt.iter = 10;
end
if ~isfield(opt,'iter')
    switch opt.ls
        case 'ADMM'
            opt.iter = 10;
        case 'CG'
        opt.iter = 1000;
    end
end
if ~exist('warping_iter','var')
    warping_iter = repmat(2, [1 numel(opt.mu)]);
end

% check parameter length
if ~((numel(warping_iter) == numel(opt.L)) && (numel(opt.L) == numel(opt.mu)))
    error('numel(warping_iter) == numel(opt.L) == numel(opt.mu) not satisfied!');
end

% get number of pyramid levels
pyramid_level = numel(warping_iter)-1;

% define scale for x and y dimensions
scale = [2 2];
dim_org = [size(img,1) size(img,2)];

% check size of L
if length(opt.L) ~= pyramid_level + 1
    error('length(opt.L) ~= pyramid_level + 1!');
end

% the warping scheme
for i = pyramid_level:-1:0 % (pyramid warping)
    img_pyramid = imresize(img, dim_org./scale.^i, 'bilinear');
    dim = size(img_pyramid(:,:,2));
    [x, y] = meshgrid(1:dim(2),1:dim(1));
    
    % pre-cache cubic spline coefficients
    c = cbanal(cbanal(img_pyramid(:,:,1)).').';

    % options
    opt_small = opt;
    opt_small.L = opt.L{i+1};
    opt_small.mu = opt.mu(i+1);
    
    % Matrix M
    M = @(x) x(1+opt_small.L(1):end-opt_small.L(1), ...
               1+opt_small.L(2):end-opt_small.L(2));
    
    for j = 1:warping_iter(i+1) % (in-level warping)
        if exist('phi','var')
            wx = imfilter(phi, [-1 1 0],  'replicate');
            wy = imfilter(phi, [-1 1 0]', 'replicate');
            x_new = -M(wx) + x;
            y_new = -M(wy) + y;
            temp_img = cat(3, cbinterp(c,y_new,x_new), img_pyramid(:,:,2));
        else
            temp_img = img_pyramid;
        end
        switch opt_small.ls     % linear solver
            case 'ADMM'
                [~, phi_delta] = main_ADMM_fast(temp_img, beta, [], opt_small);
            case 'CG'
                [~, phi_delta] = main_cg_fast(temp_img, beta, opt_small);
        end
        
        % check if the mean of phi_delta is too small; for early termination
        mean_phi = mean(abs(phi_delta(:)));
        if mean_phi < 0 % 0.1/prod(scale)^i % ( [0.314 0.628 1.257 2.094] = 2*pi./[20 10 5 3] )
            disp(['Pyr ' num2str(i) ', Warp ' num2str(j) ...
                  ', Mean of delta phi = ' num2str(mean_phi) ' < eps: ' ...
                  'Early termination'])
            if i == pyramid_level && j == 1
                phi_delta = zeros(dim+2*opt_small.L);
                phi = phi_delta;
            end
            break;
        else
            disp(['Pyr ' num2str(i) ', Warp ' num2str(j) ...
                  ', Mean of delta phi = ' num2str(mean_phi)])
        end
        
        % update phi
        if i == pyramid_level && j == 1
            phi = phi_delta;
        else
            phi = phi_delta + phi;
        end
    end
    
    if i > 0
        dim_next = scale.*dim + 2*opt.L{i};
        phi = prod( dim_next./(size(phi)) ) * imresize(phi, dim_next, 'bilinear');
    end
end

% singular case
if nnz(phi) == 0
    disp('All estimations were rejected; set output to zero')
    phi = zeros(size(phi_delta));
end

% get center part
phi = phi(1+opt.L{1}(1):end-opt.L{1}(1), 1+opt.L{1}(2):end-opt.L{1}(2));

end


% ADMM linear solver
function [x, x_full] = main_ADMM_fast(img, beta, x, opt)

%% define constants and operators

% define operators in spatial domain
nabla_x_kern = [0 0 0; -1 1 0; 0 0 0];
nabla_y_kern = [0 -1 0; 0 1 0; 0 0 0];
nabla2_kern  = [0 1 0; 1 -4 1; 0 1 0];

% define sizes
M  = size(img(:,:,1));
L1 = (size(nabla_x_kern) - 1) / 2;      L1 = L1.*opt.L;
N = M + 2*L1;

% boundary mask
M1 = @(u) cat(3, u(L1(1)+1:end-L1(1),L1(2)+1:end-L1(2),1), ...
                 u(L1(1)+1:end-L1(1),L1(2)+1:end-L1(2),2));


%% formulate the problem

% specify boundary conditions
bc = 'symmetric';     % use DCT

% define forward operators
nabla  = @(phi) cat(3, imfilter(phi, nabla_x_kern, bc), ...
    imfilter(phi, nabla_y_kern, bc));

% define adjoint operators
nablaT = @(grad_phi) imfilter(grad_phi(:,:,1), rot90(nabla_x_kern,2), bc) + ...
    imfilter(grad_phi(:,:,2), rot90(nabla_y_kern,2), bc);


%% run proximal algorithm

% initialization
if isempty(x)
    x = zeros(N);
end
zeta = zeros([N 2]);

% get the matrices
[gt, gx, gy] = partial_deriv(img);

% pre-compute
mu  = opt.mu;
gxy = gx.*gy;
gxx = gx.^2;
gyy = gy.^2;
denom = gxx + gyy + mu/2;

% store in memory at run-time
w11 = (mu/2 + gyy) ./ denom;
w12 = - gxy ./ denom;
w13 = - gx.*gt ./ denom;
w21 = (mu/2 + gxx) ./ denom;
w22 = - gxy ./ denom;
w23 = - gy.*gt ./ denom;

% proximal algorithm
if opt.isverbose
    disp('start ADMM iteration ...')
end
objval = zeros(opt.iter,1);
res    = zeros(opt.iter,1);
time   = zeros(opt.iter,1);

tic;
for k = 1:opt.iter
    % x-update
    if k == 1
        H = N(1);
        W = N(2);
        [x_coord,y_coord] = meshgrid(0:W-1,0:H-1);
        mat_x_hat = - (mu+2*beta) * ...
            (2*cos(pi*x_coord/W) + 2*cos(pi*y_coord/H) - 4);
        mat_x_hat(1) = 1;
    else
        x = idct(idct(dct(dct(mu*nablaT(u-zeta)).').' ./ mat_x_hat).').';
    end
    
    % pre-compute nabla_x
    nabla_x = nabla(x);
    
    % u-update
    u = nabla_x + zeta;
    u_temp = M1(u);
    Mu_x = u_temp(:,:,1);
    Mu_y = u_temp(:,:,2);
    
    % update u
    u(L1(1)+1:end-L1(1), L1(2)+1:end-L1(2), 1) = ...
                                            w11.*Mu_x + w12.*Mu_y + w13;
    u(L1(1)+1:end-L1(1), L1(2)+1:end-L1(2), 2) = ...
                                            w21.*Mu_y + w22.*Mu_x + w23;
    
    % zeta-update
    zeta = zeta + nabla_x - u;
    
    % record
    if opt.isverbose
        if k == 1
            % define operators
            G = @(u) gx.*u(:,:,1) + gy.*u(:,:,2);
            nabla2 = @(phi) imfilter(phi, nabla2_kern, bc);
            GT = @(Gu) cat(3, gx.*Gu, gy.*Gu);
            M1T = @(Mu) padarray(Mu, L1);
            
            % define objective function
            obj = @(phi) sum(sum(abs(G(M1(nabla(phi)))+gt).^2)) + ...
                beta * sum(sum(sum(abs(nabla(phi)).^2)));
            
            % define its gradient
            grad = @(phi) nablaT(M1T(GT(G(M1(nabla(phi)))+gt))) + ...
                2*beta*nabla2(phi);
        end
        objval(k) = obj(x);
        res(k)    = sum(sum(abs(grad(x)).^2));
        time(k)   = toc;
        disp(['ADMM iter: ' num2str(k) ...
            ', obj = ' num2str(objval(k),'%e') ...
            ', res = ' num2str(res(k),'%e')])
    end
end

% do median filtering at the last step
temp = cat(3, medfilt2( u(:,:,1) - zeta(:,:,1), [3 3] ), ...
              medfilt2( u(:,:,2) - zeta(:,:,2), [3 3] ));
x = idct(idct( dct(dct( mu*nablaT(temp) ).').' ./ mat_x_hat ).').';
if opt.isverbose
    disp(['final objective: ' num2str(obj(x),'%e')])
end
toc

% return masked x
x_full = x;
x = M1(cat(3,x,x));
x = x(:,:,1);

end


% CG linear solver
function [x, x_full] = main_cg_fast(img, beta, opt)

%% define consts and operators

% define operators in spatial domain
nabla_x_kern = [0 0 0; 1 -1 0; 0 0 0];
nabla_y_kern = [0 1 0; 0 -1 0; 0 0 0];
nabla2_kern  = -[0 1 0; 1 -4 1; 0 1 0];

% define sizes
M  = size(img(:,:,1));
L1 = (size(nabla_x_kern) - 1) / 2;      L1 = L1.*opt.L;
N = M + 2*L1;


%% formulate the problem

% specify boundary conditions
boundary_cond = 'symmetric';     % use DCT

% get the matrices
[gt, gx, gy] = partial_deriv(img);

% define forward operators
nabla  = @(phi) cat(3, imfilter(phi, nabla_x_kern, boundary_cond), ...
    imfilter(phi, nabla_y_kern, boundary_cond));

% define adjoint operators
nablaT = @(grad_phi) imfilter(grad_phi(:,:,1), rot90(nabla_x_kern,2), boundary_cond) + ...
    imfilter(grad_phi(:,:,2), rot90(nabla_y_kern,2), boundary_cond);

% define Laplacian operator
nabla2 = @(phi) imfilter(phi, nabla2_kern, boundary_cond);

% boundary masks
M1 = @(u) cat(3, u(L1(1)+1:end-L1(1),L1(2)+1:end-L1(2),1), ...
                 u(L1(1)+1:end-L1(1),L1(2)+1:end-L1(2),2));
MT = @(Mu) padarray(Mu, L1);


%% run CG

% define G and GT
G  = @(x) gx.*x(:,:,1) + gy.*x(:,:,2);
GT = @(x) cat(3, gx.*x, gy.*x);

% define A and b
A = @(x) nablaT(MT(GT(G(M1(nabla(x)))))) + beta*nabla2(x);
b = -nablaT(MT(GT(gt)));

% initialize
x = zeros(N);

% get initial r and p
r = b - A(x);
p = r;

% GGT
disp('start CG iteration ...')
objval = zeros(opt.iter,1);
res    = zeros(opt.iter,1);
time   = zeros(opt.iter,1);

tic;
for k = 1:opt.iter
    alpha = sum(abs(r(:)).^2) / sum(sum(conj(p).*A(p)));
    x = x + alpha*p;
    r_new = r - alpha*A(p);
    beta = sum(abs(r_new(:)).^2) / sum(abs(r(:)).^2);
    r = r_new;
    p = r + beta*p;

    if opt.isverbose
        if k == 1            
            % define objective function
            obj = @(phi) sum(sum(abs(G(M1(nabla(phi)))+gt).^2)) + ...
                beta * sum(sum(sum(abs(nabla(phi)).^2)));
        end
        objval(k) = obj(x);
        res(k)    = sum(abs(r(:)));
        time(k)   = toc;
        disp(['CG iter: ' num2str(k) ...
            ', obj = ' num2str(objval(k),'%e') ...
            ', res = ' num2str(res(k),'%e')])
    else
        if ~mod(k,100)
            disp(['CG iter: ' num2str(k)])
        end
    end
end

% record final objective
obj = @(phi) sum(sum(abs(G(M1(nabla(phi)))+gt).^2)) + ...
      beta * sum(sum(sum(abs(nabla(phi)).^2)));
disp(['final objective: ' num2str(obj(x),'%e')])
toc

% return masked x
x_full = x;
x = M1(cat(3,x,x));
x = x(:,:,1);

end


% function to get gx, gy and gt
function [It,Ix,Iy] = partial_deriv(images)

% derivative kernel
h = [1 -8 0 8 -1]/12;

% blending ratio
b = 0.5;    

% get images
img1 = images(:,:,1);
img2 = images(:,:,2);

% get gt
It = images(:,:,2) - images(:,:,1);

% first compute derivative then warp
I2x = imfilter(img2, h,  'replicate');
I2y = imfilter(img2, h', 'replicate');

% temporal average
I1x = imfilter(img1, h,  'replicate');
I1y = imfilter(img1, h', 'replicate');

% get gx and gy
Ix  = b*I2x + (1-b)*I1x;
Iy  = b*I2y + (1-b)*I1y;

end


% cubic spline interpolation function (backward: coeffcient -> image)
function out = cbinterp(c,x,y)

% calculate movements
px = floor(x);
fx = x - px;
py = floor(y);
fy = y - py;

% define device functions
w0 = @(a) (1/6)*(a.*(a.*(-a + 3) - 3) + 1);
w1 = @(a) (1/6)*(a.*a.*(3*a - 6) + 4);
w2 = @(a) (1/6)*(a.*(a.*(-3*a + 3) + 3) + 1);
w3 = @(a) (1/6)*(a.*a.*a);
r  = @(x,c0,c1,c2,c3) c0.*w0(x) + c1.*w1(x) + c2.*w2(x) + c3.*w3(x);

% define texture function
tex = @(x,y) interp2(c,y,x,'nearest',0);

% elementwise lookup
out = r(fy, ...
      r(fx, tex(px-1,py-1), tex(px,py-1), tex(px+1,py-1), tex(px+2,py-1)), ...
      r(fx, tex(px-1,py  ), tex(px,py  ), tex(px+1,py  ), tex(px+2,py  )), ...
      r(fx, tex(px-1,py+1), tex(px,py+1), tex(px+1,py+1), tex(px+2,py+1)), ...
      r(fx, tex(px-1,py+2), tex(px,py+2), tex(px+1,py+2), tex(px+2,py+2)));
end


% cubic spline interpolation function (forward: image -> coeffcient)
function c = cbanal(img)

[m,n] = size(img);

% A = toeplitz([4 1 zeros(1,m-2)]) / 6;

c = zeros(m,n);
for i = 1:n
%     c(:,i) = A \ y(:,i);
    c(:,i) = 6 * tridisolve(img(:,i));
end

end


% triangular linear solver
function x = tridisolve(d)

% initialize
x = d;

% get length
m = length(x);

% define a, b and c
b = 4*ones(m,1);

% forward
for j = 1:m-1
    mu = 1/b(j);
    b(j+1) = b(j+1) - mu;
    x(j+1) = x(j+1) - mu*x(j);
end

% backward
x(m) = x(m)/b(m);
for j = m-1:-1:1
    x(j) = (x(j)-x(j+1))/b(j);
end

end

