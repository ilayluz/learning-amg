function [rows, cols, values] = drop_row_col(A_rows, A_cols, A_values, total_size, indices)
total_size = double(total_size);

A = sparse(A_rows, A_cols, A_values, total_size, total_size);
A(indices, :) = [];
A(:, indices) = [];
[rows, cols, values] = find(A);
end