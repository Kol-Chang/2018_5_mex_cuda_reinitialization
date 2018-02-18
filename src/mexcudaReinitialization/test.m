%
mexcuda mexcudaReinitialization.cu Reinitialization.cu

% test mexcudaReinitialization scheme

	% 
		a = 210;
		b = 210;
		c = 114;

		xv = linspace(-250,250,64);
		yv = xv;
		%zv = xv(abs(xv)<150);
		zv = xv;

		[x, y, z] = meshgrid(xv, yv, zv); % simulation domain in nm
		dx = xv(2)-xv(1);
		dy = yv(2)-yv(1);
		dz = zv(2)-zv(1);

		F = sqrt(x.^2/a^2 + y.^2/b^2 + z.^2/c^2) - 1;

%

new_F = mexcudaReinitialization(F,[dx, dy, dz]);