function [U, phi] = sdoss_gen_scalarU(Udim, phaserange, k, method)

% Note: phaserange is measured in terms of wavelength!

switch k
    case 'unitary'
        phase = zeros(Udim);
    case 'tilt'
        rng(0);
        A = zeros(Udim);
        A(randperm(Udim(1),1),randperm(Udim(2),1)) = 1;
        U = ifft2(ifftshift(A));
        phase = unwrapMultipart(angle(U));
    case 'gaussian'
%         sigma = 1000;
        sigma = 500;
        phase = gaussiann(Udim,sigma);
    case 'Zernike'
        a = [1 0 0 3 5 2 2]';   % Zernike coefficients
        M = length(a);                    % number of Zernike orders
        [~, Z] = zernike_coeffs(zeros(Udim), M);
        phase = reshape(Z*a, Udim);       % construct phase using Zernike
    otherwise
        error('scalar type undefined')
end
phi = normphase(phase, phaserange);
phi = phi - mean2(phi);
U = ones(Udim).*exp(1i*2*pi*phi);

switch method
    case 'DI'
        return
    case 'AS'
        U = padarray(U, Udim);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

end

function angleout = normphase(anglein, phaserange)
anglein = anglein - min(anglein(:));
if max(anglein(:)) == 0
    angleout = anglein;
else
    angleout = phaserange*anglein/max(anglein(:));
end

end

function f = gaussiann(Udim,sigma)

switch numel(Udim)
    case 2
        X = linspace(round(-Udim(1)/2), round(Udim(1)/2), Udim(1));
        Y = linspace(round(-Udim(2)/2), round(Udim(2)/2), Udim(2));
        [y, x] = meshgrid(Y, X);
        f = exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
    case 3
        X = linspace(round(-Udim(1)/2), round(Udim(1)/2), Udim(1));
        Y = linspace(round(-Udim(2)/2), round(Udim(2)/2), Udim(2));
        Z = linspace(round(-Udim(3)/2), round(Udim(3)/2), Udim(3));
       [y, x, z] = meshgrid(Y, X, Z);
        f = exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2)-z.^2/(2*sigma^2));
    otherwise
        error('the input dimension is undefined')
end
f = f./sum(f(:));

end

