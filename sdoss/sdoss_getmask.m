function intensity_mask = sdoss_getmask(tA, factor, method)

switch method
    case 'DI'
        intensity_mask = tA(1:factor:end,1:factor:end).^2;
    case 'AS'
        intensity_mask = diff_crop(tA(1:factor:end,1:factor:end)).^2;
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

    function A = diff_crop(B)
        pdim = size(B)/3;
        A = B(pdim(1)+1:end-pdim(1),pdim(2)+1:end-pdim(2));
    end

end