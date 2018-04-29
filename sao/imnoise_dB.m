function I_noisy = imnoise_dB(I, SNR)

% Adjust intensities in image I to range from 0 to 1
min_I = min(I(:));
max_I = max(I(:));
I = I - min_I;
I = I / (max_I - min_I);

% Add noise to image
v = var(I(:)) / 10^(SNR/20);
I_noisy = imnoise(I, 'gaussian', 0, v);
I_noisy = I_noisy * (max_I - min_I) + min_I;
