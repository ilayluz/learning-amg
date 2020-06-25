function [rows, cols, values] = drop_zero_row_col(A_rows, A_cols, A_values, total_size)
total_size = double(total_size);

A = sparse(A_rows, A_cols, A_values, total_size, total_size);
A(~any(A,2), :) = [];
A(:, ~any(A,1)) = [];
[rows, cols, values] = find(A);
end