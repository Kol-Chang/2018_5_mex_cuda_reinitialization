addpath(genpath('..'))

xv = linspace(-5,5,64);
yv = xv;
zv = xv;

dx = xv(2) - xv(1);
dy = yv(2) - yv(1);
dz = zv(2) - zv(1);

[x, y, z] = meshgrid(xv,yv,zv);

fun = @(x,y,z) (0.1+(x-3.5).^2+(sqrt(y.^2+z.^2)-2).^2) .* (sqrt(x.^2/4+(z.^2+y.^2)/9)-1);

F = fun(x,y,z);

F_g = gpuArray(F);
%ds = gpuArray([dx,dy,dz]);
ds = [dx,dy,dz];

xpr = zeros(size(F),'gpuArray');
ypf = xpr;
zpu = xpr;

F_cur = F_g;

nel = product(size(F));
[rows,cols,pages] = size(F);



