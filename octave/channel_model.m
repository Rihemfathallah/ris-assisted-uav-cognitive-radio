function [h_d, a] = channel_model(p, nh, nv, psi)
  % Independent Octave implementation of the SU-side single-antenna channel
  % (src/ris_cr/channel.py). Returns noise-normalised h_d and cascaded a = h_r .* g.
  %   RIS: UPA in the y-z plane, a_RIS = kron(a_z, a_y); direction cosines;
  %   LoS includes exp(-j 2 pi d / lambda); Rician with unit-power NLoS.
  uav = [p.radius_m*cos(psi); p.radius_m*sin(psi); p.height_m];
  N = nh*nv;
  ula = @(n, u) exp(1j*pi*(0:n-1).' * u);
  upa = @(u) kron(ula(nv, u(3)), ula(nh, u(2)));
  cn  = @(varargin) (randn(varargin{:}) + 1j*randn(varargin{:}))/sqrt(2);
  ric = @(pl, KdB, los, nl) sqrt(pl) * (sqrt(10^(KdB/10)/(10^(KdB/10)+1))*los + sqrt(1/(10^(KdB/10)+1))*nl);
  C0 = 10^(p.C0_dB/10);
  % UAV -> RIS
  v = uav - p.ris;  d = norm(v);  u = v/d;
  g = ric(C0*d^-p.alpha_uav_ris, p.K_uav_ris_dB, exp(-2j*pi*d/p.lambda)*upa(u), cn(N,1));
  % RIS -> SU
  v = p.su - p.ris;  d = norm(v);  u = v/d;
  h_r = ric(C0*d^-p.alpha_ris_ground, p.K_ris_ground_dB, exp(-2j*pi*d/p.lambda)*upa(u), cn(N,1));
  % UAV -> SU (direct)
  d = norm(p.su - uav);
  pl = C0*d^-p.alpha_uav_ground * 10^(-p.blockage_dB/10);
  h_d = ric(pl, p.K_uav_ground_dB, exp(-2j*pi*d/p.lambda), cn(1,1));
  s = sqrt(p.sigma2_W);
  h_d = h_d / s;
  a = (h_r .* g) / s;
end
