% Octave replication of Fig. 1 (rate vs N, no PU constraint, P = P_max).
%  (A) exact check: rates recomputed from the realisations exported by Python
%      (results/octave/fig1_input_N*.mat) and compared element-wise;
%  (B) independent check: Octave's own channel generator (channel_model.m),
%      2000 i.i.d. snapshots per N, mean rates written to CSV.
% Run from the repository root:  octave-cli octave/rate_vs_N.m
addpath('octave');
p = params();
rand('seed', 1); randn('seed', 1);
Ns = [8 16 32 64];  shapes = [4 2; 4 4; 8 4; 8 8];
fid = fopen('results/octave/rate_vs_N_octave.csv', 'w');
fprintf(fid, 'N,method,rate,rate_sem,max_abs_diff_vs_python\n');
for k = 1:numel(Ns)
  N = Ns(k);
  S = load(sprintf('results/octave/fig1_input_N%d.mat', N));
  E = exp(1j*S.theta);                                  % (B x N)
  r_no   = log2(1 + S.P*abs(S.h_d).^2);
  r_rand = log2(1 + S.P*abs(S.h_d + sum(S.a .* E, 2)).^2);
  r_cf   = log2(1 + S.P*(abs(S.h_d) + sum(abs(S.a), 2)).^2);
  dno = max(abs(r_no - S.py_no)); drand = max(abs(r_rand - S.py_rand)); dcf = max(abs(r_cf - S.py_cf));
  printf('N=%2d exact check: max|diff| no_ris %.2e  random %.2e  closed_form %.2e\n', N, dno, drand, dcf);
  B = numel(r_no);
  fprintf(fid, '%d,no_ris,%.10f,%.10f,%.3e\n', N, mean(r_no), std(r_no)/sqrt(B), dno);
  fprintf(fid, '%d,random,%.10f,%.10f,%.3e\n', N, mean(r_rand), std(r_rand)/sqrt(B), drand);
  fprintf(fid, '%d,closed_form,%.10f,%.10f,%.3e\n', N, mean(r_cf), std(r_cf)/sqrt(B), dcf);
  % (B) independent generator
  M = 2000; rc = zeros(M,1); rr = zeros(M,1); rn = zeros(M,1);
  for m = 1:M
    [h_d, a] = channel_model(p, shapes(k,1), shapes(k,2), 2*pi*rand());
    rc(m) = log2(1 + p.P_max_W*(abs(h_d) + sum(abs(a)))^2);
    rr(m) = log2(1 + p.P_max_W*abs(h_d + a.'*exp(2j*pi*rand(N,1)))^2);
    rn(m) = log2(1 + p.P_max_W*abs(h_d)^2);
  end
  printf('N=%2d independent MC: closed_form %.4f +- %.4f\n', N, mean(rc), std(rc)/sqrt(M));
  fprintf(fid, '%d,closed_form_iid_octave,%.10f,%.10f,nan\n', N, mean(rc), std(rc)/sqrt(M));
  fprintf(fid, '%d,random_iid_octave,%.10f,%.10f,nan\n', N, mean(rr), std(rr)/sqrt(M));
  fprintf(fid, '%d,no_ris_iid_octave,%.10f,%.10f,nan\n', N, mean(rn), std(rn)/sqrt(M));
end
fclose(fid);
