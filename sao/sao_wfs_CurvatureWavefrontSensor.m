function [I1, I0, I2] = sao_wfs_CurvatureWavefrontSensor(U0, pdim, sampling_rate, lambda, z, SNR)

% get I0
I0 = sdoss_sensor(U0, pdim, 'AS');

% propagate the field
U1 = sdoss_propagation(U0, lambda, sampling_rate, -z, 'AS');
I1 = sdoss_sensor(U1, pdim, 'AS');
U2 = sdoss_propagation(U0, lambda, sampling_rate, z, 'AS');
I2 = sdoss_sensor(U2, pdim, 'AS');

c = 255 / mean2(I0);

% add noise
I0 = imnoise_dB(c*I0, SNR)/c;
I1 = imnoise_dB(c*I1, SNR)/c;
I2 = imnoise_dB(c*I2, SNR)/c;

% % define ADC
adc = @(I) round(I .* c) ./ c;

% convert
I0 = adc(I0);
I1 = adc(I1);
I2 = adc(I2);
