function wavefront = sdoss_getwavefront(U, factor, method)

switch method
    case 'DI'
        wavefront = scalar_inte(unwrapMultipart(angle(U),'radians'),factor);
    case 'AS'
        wavefront = scalar_inte(unwrapMultipart(angle(diff_crop(U)),'radians'),factor);
    otherwise
        error('Undefined method. Method can only be either DI or AS');
end

    function A = diff_crop(B)
        pdim = size(B)/3;
        A = B(pdim(1)+1:end-pdim(1),pdim(2)+1:end-pdim(2));
    end

    function A = scalar_inte(B, factor)
        A = zeros(size(B)/factor);
        for i = 1:factor
            for j = 1:factor
                A = A + B(i:factor:end,j:factor:end);
            end
        end
    end

end
