function img_capture = sao_wfs_CodedWavefrontSensor(U0, tA, pdim, sampling_rate, ...
                                                            lambda, z, SNR)
U1 = sdoss_aperture(U0, tA);
U  = sdoss_propagation(U1, lambda, sampling_rate, z, 'AS');
img_capture = sdoss_sensor(U, pdim, 'AS');

c = 255 / mean2(img_capture);

% add noise
img_capture = imnoise_dB(c*img_capture, SNR)/c;

% % define ADC
adc = @(I) round(I .* c) ./ c;

% convert
img_capture = adc(img_capture);
