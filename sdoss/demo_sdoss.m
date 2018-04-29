clc;close all;clear;

%% Parameters
% sensor pixels
pdim = [128 128];
pixel_size = 5e-6;
factor = 10;     % factor = pixel_size/sampling_rate
z = 1e-6;
wavefront_range = 10;
lambda = 550e-9;

% scalar field
Udim = factor*pdim;
sampling_rate = pixel_size/factor;

% mask
level = 1;

sdoss_para_check(pdim, pixel_size, factor, lambda, z, 'DI');
sdoss_para_check(pdim, pixel_size, factor, lambda, z, 'AS');

%% SDOSS AS imaging
tic;
tA_AS = sdoss_gen_mask(pdim, factor, 'rand', level, [], 'AS');
U0_AS = sdoss_gen_scalarU(Udim, wavefront_range, 'unitary', 'AS');
U1_AS = sdoss_aperture(U0_AS, tA_AS);
U_AS  = sdoss_propagation(U1_AS, lambda, sampling_rate, z, 'AS');

img_AS = sdoss_sensor(U_AS, pdim, 'AS');

mask_AS = sdoss_getmask(tA_AS, factor, 'AS');

figure;     imshow(img_AS, []);      title('image\_AS');
toc

%% SDOSS DI imaging
tic;
tA_DI = sdoss_gen_mask(pdim, factor, 'rand', level, [], 'DI');
U0_DI = sdoss_gen_scalarU(Udim, wavefront_range, 'unitary', 'DI');
U1_DI = sdoss_aperture(U0_DI, tA_DI);
U_DI  = sdoss_propagation(U1_DI, lambda, sampling_rate, z, 'DI');

img_DI = sdoss_sensor(U_DI, pdim, 'DI');

mask_DI = sdoss_getmask(tA_DI, factor, 'DI');

figure;     imshow(img_DI,[]);      title('image\_DI');
toc

%% Compare
disp(['relative error is ' num2str(norm(img_DI-img_AS) / ...
                       mean(norm(img_DI)+norm(img_AS)))])

