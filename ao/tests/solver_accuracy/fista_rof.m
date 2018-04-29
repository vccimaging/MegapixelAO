function [tex, struct] = fista_rof(im, theta, nIters, alp)
% Decompose the input IMAGE into structure and texture parts using the
% Rudin-Osher-Fatemi method (L1 denoising by FISTA). The final output is a
% linear combination of the decomposed texture and the structure parts.

if nargin == 1
    theta  = 1/8;
    nIters = 100;
    alp    = 0.95;  % alp = 0.75 results in 4:1
end

IM = scale_image(im, -1,1);
im = IM;

IM = cat(3, fista_l1_denoising(IM(:,:,1), theta, nIters), ...
            fista_l1_denoising(IM(:,:,2), theta, nIters) );

tex = squeeze(scale_image(im - alp*IM, 0, 255));

if nargout == 2
    struct = squeeze(scale_image(IM, 0, 255)); %(u-min(u(:)))/(max(u(:))-min(u(:))) - 1;
end

end



function x = fista_l1_denoising(b, theta, iter)

% set boundary condition
boundary_cond = 'replicate';

% set kernel
x_kern = [0 0 0; 1 -1 0; 0 0 0];
y_kern = [0 1 0; 0 -1 0; 0 0 0];

% define operators
nabla  = @(x) cat(3, imfilter(x, x_kern, boundary_cond), ...
                     imfilter(x, y_kern, boundary_cond));
nablaT = @(nabla_x) imfilter(nabla_x(:,:,1), rot90(x_kern,2), boundary_cond) + ...
                    imfilter(nabla_x(:,:,2), rot90(y_kern,2), boundary_cond);

% define proximal operator of ||x||_{\infty} <= 1
prox_linf = @(x) min(max(x, -1) ,1);

% define objective
obj = @(p) sum(sum(abs(nablaT(p)- b/theta).^2));

% set step size tk
tau = 1/4;

% initialization
p = zeros([size(b) 2]);

% proximal gradient descent for dual problem
tt = zeros(iter,1);
for k = 1:iter
    tt(k) = obj(p);
    disp(['k = ' num2str(k) ', obj = ' num2str(obj(p),'%e')]);
    p = prox_linf(p - tau*nabla(nablaT(p) - b/theta));
end

% get primal solution
x =  b - nablaT(p)*theta;

end



function imo = scale_image(im, vlow, vhigh, ilow, ihigh)
% linearly scale the input image IM to the range [vlow, vhigh]
% from the range [ilow ihigh] (if provided)

if nargin == 3
    ilow  = min(im(:));
    ihigh = max(im(:));
end

imo = (im-ilow)/(ihigh-ilow) * (vhigh-vlow) + vlow;

end
