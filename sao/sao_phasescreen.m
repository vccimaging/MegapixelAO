function phz = sao_phasescreen(r0, N, delta, L0, l0)
% Generate phase screen using the sub-harmonic method: 
% 
% R.G. Lane, A. Glindemann, and J. C. Dainty. "Simulation of a Kolmogorov
% phase screen." Waves in random media 2.3 (1992): 209-224. 
% 
% The output phase is in units of wavelengths.

% aperture size
D = N*delta;

% high-frequency screen from FFT method
phz_hi = ft_phase_screen(r0, N, delta, L0, l0);

% spatial grid [m]
[x, y] = meshgrid((-N/2 : N/2-1) * delta);

% initialize low-freq screen
phz_lo = zeros(size(phz_hi));

% loop over frequency grids with spacing 1/(3^p*L)
for p = 1:3
    
    % setup the PSD
    del_f = 1 / (3^p*D); %frequency grid spacing [1/m]
    fx = (-1 : 1) * del_f;
    
    % frequency grid [1/m]
    [fx, fy] = meshgrid(fx);
    f = sqrt(fx.^2 + fy.^2);
    fm = 5.92/l0/(2*pi); % inner scale frequency [1/m]
    f0 = 1/L0; % outer scale frequency [1/m]

    % modified von Karman atmospheric phase PSD
    PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ./ (f.^2 + f0^2).^(11/6);
    PSD_phi(2,2) = 0;
    
    % random draws of Fourier coefficients
    cn = (randn(3) + 1i*randn(3)) .* sqrt(PSD_phi)*del_f;
    SH = zeros(N);
    
    % loop over frequencies on this grid
    for ii = 1:9
        SH = SH + cn(ii) * exp(1i*2*pi*(fx(ii)*x+fy(ii)*y));
    end
    
    phz_lo = phz_lo + SH; % accumulate subharmonics
end
phz_lo = real(phz_lo) - mean2(real(phz_lo));

% final phase (normalized in wavelengths)
phz = (phz_lo + phz_hi) / (2*pi);

end



function phz = ft_phase_screen(r0, N, delta, L0, l0)
% Generate phase screen using the FFT method

% setup the PSD
del_f = 1/(N*delta); % frequency grid spacing [1/m]
fx = (-N/2 : N/2-1) * del_f;

% frequency grid [1/m]
[fx, fy] = meshgrid(fx);
f = sqrt(fx.^2 + fy.^2);
fm = 5.92/l0/(2*pi); % inner scale frequency [1/m]
f0 = 1/L0; % outer scale frequency [1/m]

% modified von Karman atmospheric phase PSD
PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ./ (f.^2 + f0^2).^(11/6);
PSD_phi(N/2+1,N/2+1) = 0;

% random draws of Fourier coefficients
cn = (randn(N) + 1i*randn(N)) .* sqrt(PSD_phi)*del_f;

% synthesize the phase screen
phz = real(ifftshift(ifft2(fftshift(cn))*numel(cn)));

end

