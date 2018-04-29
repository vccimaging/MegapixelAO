clc;clear;close all;
%%%% Warning: Please run it on a large memory PC! 

% check memory
mem = memory;
if (mem.MemAvailableAllArrays / 1e9) < 8
    error('Memory < 8 GB; Please run this code on a larger memory PC!')
end

% add path
addpath('../')
addpath('../../sdoss');
addpath('../../solvers/');
addpath('../../utilities/');

% to be reproducible
rng('default')

% set parameters
pdim = [512 512];                   % sensor spatial resolution
pixel_size = 6.45e-6;               % sensor pixel size
factor = 5;                         % upsampling factor
z = 100e-6;                         % distance between mask and sensor
lambda = 550e-9;                    % wavelenth

% get parameters of the scalar field
Udim = factor*pdim;
sampling_rate = pixel_size/factor;

% set parameters for lenslets
f = 7e-3;                               % effective focal length
numLens = [32 32];                      % number of lenslets
pdim_each = pdim./numLens;
D = pdim_each(1)*pixel_size;            % aperture width

% set noise
SNR = 30;   % set signal-to-noise ratio (in dB)

% calibration of Shack-Hartmann
[img_SH_ref, x_org, y_org] = sao_wfs_shackhartmann(ones(factor*pdim), numLens, D, lambda, ...
                                f, pdim_each, factor, SNR);

% simulate wavefront
wavefront_range = 10;

% generate sample scalar field
[U0_DI, phi_GT] = sdoss_gen_scalarU(factor*pdim, wavefront_range, 'gaussian', 'DI');

% run sao Shack-Hartmann simulator
[img_SH, x_peak, y_peak] = sao_wfs_shackhartmann(U0_DI, numLens, D, lambda, ...
                                f, pdim_each, factor, SNR);

% show sensor image
figure;     imshow(cell2mat(img_SH_ref),[]);        title('reference image');
figure;     imshow(cell2mat(img_SH),[]);
title(['captured image, wavefront range = ' num2str(wavefront_range) '\lambda']);

% wavefront slopes
Sx = (x_peak - x_org) * pixel_size^2*pdim_each(1) / (lambda*f);
Sy = (y_peak - y_org) * pixel_size^2*pdim_each(2) / (lambda*f);

% reconstruct wavefront
phi_SH = poisson_solver(Sy,Sx);

% normarlize
phi_SH = phi_SH - min(phi_SH(:));
phi_GT = imresize(phi_GT, 1/factor);
phi_GT = phi_GT - min(phi_GT(:));

% show results
map = coolwarm(256);
mydisplay(phi_GT,  wavefront_range, map);    title('ground truth');
mydisplay(imresize(phi_SH,pdim),wavefront_range, map);
title('Shack-Hartmann (interpolated)');

