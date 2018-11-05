function writeImage(filename, img, cmap)
%WRITE AN IMAGE TO PNG
%
% Writes the image 'img' to a png-file with name 'filename' and the 
% specified colormap.

if nargin < 3
    cmap = gray(256);
end

imwrite(255*img, cmap, [filename '.png']) 
    
end