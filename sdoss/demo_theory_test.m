clc;clear;close all;
% This theory test demo reproduces Fig 4 in paper:
% [*] Fabin Shen and Anbo Wang. "Fast-Fourier-transform based numerical 
% integration method for the Rayleigh-Sommerfeld diffraction formula." 
% Applied optics 45.6 (2006): 1102-1110.

%% Circular aperture test
N = 513;
lambda = 0.5e-6;        k = 2*pi/lambda;
a = 10*lambda;
sampling_rate = 0.1*lambda;
U0_DI = sdoss_gen_scalarU([N N], 1, 'unitary', 'DI');
mask_DI = sdoss_gen_mask([N N], 1, 'circular', [], a/sampling_rate, 'DI');
U1 = sdoss_aperture(U0_DI, mask_DI);

z_vec = 1e-6:2e-7:10e-6;
Iz = z_vec.*( exp(1i*k*z_vec)./z_vec - ...
              exp(1i*k*sqrt(z_vec.^2+a^2))./sqrt(z_vec.^2+a^2) );

I_FFTDI = zeros(numel(z_vec),1);
I_FFTAS = zeros(numel(z_vec),1);
cc = 0;
for z = z_vec
    cc = cc + 1;
    disp([num2str(cc) '/' num2str(numel(z_vec))])
    
%     sdoss_para_check([N N], sampling_rate, 1, lambda, z, 'DI')
%     sdoss_para_check([N N], sampling_rate, 1, lambda, z, 'AS')

    U_FFTDI = sdoss_propagation(U1, lambda, sampling_rate, z, 'DI');
    I_FFTDI(cc) = U_FFTDI(round(size(U_FFTDI,1)/2),round(size(U_FFTDI,2)/2));

    U_FFTAS = sdoss_propagation(U1, lambda, sampling_rate, z, 'AS');
    I_FFTAS(cc) = U_FFTAS(round(size(U_FFTAS,1)/2),round(size(U_FFTAS,2)/2));
end
figure;
plot(z_vec, [abs(Iz').^2 abs(I_FFTDI).^2 abs(I_FFTAS).^2], '-.o');
legend('Theory', 'DI result', 'AS result');


%% Single Slit test
clear
N = 1025;
pdim = [N N];
lambda = 0.5e-6;
width = 10e-6;
sampling_rate = 80e-6/N;
z = 100e-6;
x = (1:N) - (N+1)/2;

sdoss_para_check(pdim, sampling_rate, 1, lambda, z, 'DI')
sdoss_para_check(pdim, sampling_rate, 1, lambda, z, 'AS')

% DI single slit
U0_DI = sdoss_gen_scalarU(pdim, 1, 'unitary', 'DI');
mask_DI = sdoss_gen_mask(pdim, 1, 'square', [], width/sampling_rate, 'DI');
U1_DI = sdoss_aperture(U0_DI, mask_DI);
U2_DI = sdoss_propagation(U1_DI, lambda, sampling_rate, z, 'DI');
img_DI = sdoss_sensor(U2_DI, pdim, 'DI');

% AS single slit
U0_AS = sdoss_gen_scalarU(pdim, 1, 'unitary', 'AS');
mask_AS = sdoss_gen_mask(pdim, 1, 'square', [], width/sampling_rate, 'AS');
U1_AS = sdoss_aperture(U0_AS, mask_AS);
U2_AS = sdoss_propagation(U1_AS, lambda, sampling_rate, z, 'AS');
img_AS = sdoss_sensor(U2_AS, pdim, 'AS');

% show results
figure;     plot(x*sampling_rate, [sum(img_DI)' sum(img_AS)']);
legend('DI simulation result', 'AS simulation result');
figure;     imshow(img_DI, []);     title('DI simulation result');
figure;     imshow(img_AS, []);     title('AS simulation result');

