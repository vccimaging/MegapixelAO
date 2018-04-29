function sdoss_para_check(pdim, pixel_size, factor, lambda, z, method)

if numel(pdim)~=2
    error('pdim should be a 2 element matrix');
end

if pixel_size < 0
    error('pixel_size should be a positive number')
end

if factor < 1
    error('factor should be a positive integer larger or equal to 1')
elseif factor ~= round(factor)
    warning('factor should be an integer')
end

if lambda < 0
    error('lambda should be a positive number')
end

if z < 0
    error('z should be a positive number')
end

switch method
    case 'DI'
        N = pdim * factor;
        delta_x = pixel_size * pdim;
        N_min = delta_x / lambda * 2 .* sin(atan(delta_x/z));
        if sum(N < N_min)
            disp('DI method: Failed the exam. Kernel undersampled.');
            disp(['N(1) = ' num2str(N(1)) ' < N_min(1) = ' num2str(N_min(1))]);
            disp(['N(2) = ' num2str(N(2)) ' < N_min(2) = ' num2str(N_min(2))]);
            disp('DI method: N should be larger than N_min.');
            disp('DI method: Please re-tune your parameters.');
        else
            disp('DI method: Pass parameter checking exam!')
            disp(['N(1) = ' num2str(N(1)) ' > N_min(1) = ' num2str(N_min(1))]);
            disp(['N(2) = ' num2str(N(2)) ' > N_min(2) = ' num2str(N_min(2))]);
        end
    case 'AS'
        disp('AS method: Pass parameter checking exam!')
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end
