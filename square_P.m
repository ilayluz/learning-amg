function [rows, cols] = square_P(P_rows, P_cols, P_values, total_size, coarse_nodes)
total_size = double(total_size);
P_num_rows = total_size;
[~, P_num_cols] = size(coarse_nodes);

P = sparse(P_rows, P_cols, P_values, P_num_rows, P_num_cols);
P_square = sparse(total_size, total_size);
P_square(:, coarse_nodes) = P;
[rows, cols] = find(P_square);
end

