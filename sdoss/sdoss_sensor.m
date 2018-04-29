function img = sdoss_sensor(U, pdim, method)

switch method
    case 'DI'
        img = FFT_DI_sensor(U, pdim);
    case 'AS'
        img = diff_sensor(U, pdim);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

    function img = FFT_DI_sensor(U, pdim)
        
        temp_img = abs(U).^2;
        factor = round(size(temp_img)/pdim);
        img = zeros(pdim);
        for i = 1:factor
            for j = 1:factor
                img = img + temp_img(i:factor:end,j:factor:end);
            end
        end
        
    end

    function img = diff_sensor(U, pdim)
        
        temp_img = abs(U).^2;
        Usize = size(temp_img)/3;
        temp_img = temp_img(Usize(1)+1:end-Usize(1),Usize(2)+1:end-Usize(2));
        
        factor = round(size(temp_img)/pdim);
        img = zeros(pdim);
        for i = 1:factor
            for j = 1:factor
                img = img + temp_img(i:factor:end,j:factor:end);
            end
        end
        
    end

end
