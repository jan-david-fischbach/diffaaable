clear
clc

function result = f_test_harder(z, residues, n_poles)
    poles = (0.7 + 0.3i) .^ (0:n_poles-1);
    disp(['poles = ', mat2str(poles)]);
    
    % Compute the summation
    denom = z(:) - reshape(poles, [1, 1, n_poles]);
    residues = reshape(residues.', [1, size(residues,2), size(residues,1)]);
    size(denom)

    result = sum(residues ./ denom, 3); % Sum over the last dimension
end

z_k = linspace(-4, 4, 1000) + 0.8i;
n_poles = 11;
residues = reshape(0:n_poles*300-1, n_poles, []);
f_k = f_test_harder(z_k, residues, 11);


[r, pol, res] = set_aaa(f_k, z_k);
pol
res