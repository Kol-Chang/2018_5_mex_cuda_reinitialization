%
%mexcuda mexcudaReinitialization.cu Reinitialization.cu

addpath(genpath('..'))

% test mexcudaReinitialization scheme

% 
xv = linspace(-5,5,64);
yv = xv;
zv = xv;

[x, y, z] = meshgrid(xv,yv,zv);

fun = @(x,y,z) (0.1+(x-3.5).^2+(sqrt(y.^2+z.^2)-2).^2) .* (sqrt(x.^2/4+(z.^2+y.^2)/9)-1);

F = fun(x,y,z);


for i=1:1
	new_F = mexcudaReinitialization(F,[dx, dy, dz]);
end

