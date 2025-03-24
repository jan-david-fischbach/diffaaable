clear
clc

function result = f_test_harder(z, residues, n_poles)
    poles = (0.1) .^ (0:n_poles-1);
    disp(['poles = ', mat2str(poles)]);
    
    % Compute the summation
    denom = z(:) - reshape(poles, [1, 1, n_poles]);

    residues = reshape(residues, [1, size(residues,1), size(residues,2)]);

    result = sum(residues ./ denom, 3); % Sum over the last dimension
end

z_k = linspace(-4, 4, 15) + 0.8i;
n_poles = 3;
residues = reshape(0:n_poles*2-1, [], n_poles);
f_k = f_test_harder(z_k, residues, n_poles);

format longE 

[r, pol, res, zer, z, f, w, errs] = set_aaa(f_k, z_k);

errs = errs(1: length(z))