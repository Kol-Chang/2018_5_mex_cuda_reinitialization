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

nel = prod(size(F));
[rows,cols,pages] = size(F);

b_c = parallel.gpu.CUDAKernel('Ke.ptx','Ke.cu','boundary_correction');
tsl = parallel.gpu.CUDAKernel('Ke.ptx','Ke.cu','time_step_lsf');

ThreadBlockSize = [rows,4,1];
GridSize = [1,16,64];

b_c.ThreadBlockSize = ThreadBlockSize;
b_c.GridSize = GridSize;

tsl.ThreadBlockSize = ThreadBlockSize;
tsl.GridSize = GridSize;



