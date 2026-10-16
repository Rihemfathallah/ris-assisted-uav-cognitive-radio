function p = params()
  % Mirror of configs/default.yaml (checked by tests/test_octave_params.py).
  p.P_max_dBm = 20;
  p.B_Hz = 1e6;
  p.NF_dB = 10;
  p.C0_dB = -30;
  p.alpha_uav_ris = 2.2;  p.alpha_ris_ground = 2.8;  p.alpha_uav_ground = 3.6;
  p.K_uav_ris_dB = 10;    p.K_ris_ground_dB = 3;     p.K_uav_ground_dB = 0;
  p.blockage_dB = 0;
  p.height_m = 100;  p.radius_m = 30;
  p.ris = [60; 0; 10];  p.su = [65; 5; 0];  p.pu = [55; -10; 0];
  p.sigma2_W = 10^((-174 + 10*log10(p.B_Hz) + p.NF_dB - 30)/10);
  p.P_max_W  = 10^((p.P_max_dBm - 30)/10);
  p.lambda   = 4*pi*sqrt(10^(p.C0_dB/10));
end
