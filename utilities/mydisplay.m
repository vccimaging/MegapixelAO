function mydisplay(Z,maxZ,yourcolormap)

if isempty(maxZ)
    disp('maxZ is not defined');
    maxZ = max(Z(:));
end

if isempty(yourcolormap)
    disp('colormap is not defined');
    yourcolormap = gray;
end

% show height map
figure;     surf(Z,mod(Z,1));view([-45 15])
shading interp;colormap(yourcolormap);axis tight;axis off;
zlim([0 maxZ]);
