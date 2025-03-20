clear
clc

z_k = -1:0.5:1;
f_k = [2./(z_k-2j) + 2./(z_k-1j+0.1) + 2./(z_k-3); 
       1j./(z_k-2j) + 1j./(z_k-1j+0.1) + 1j./(z_k-3)].';

[r, pol, res] = set_aaa(f_k, z_k, 'mmax', 5);
pol
res