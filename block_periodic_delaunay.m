function [A, points] = block_periodic_delaunay(k,b)
% Create b by b block - doubly periodic triangulation with k vertices per
% block. Put random lognormal coefficients on the edges and create graph
% Laplacian.
if (b < 3)
	disp('WARNING: b cannot be less than 3, resetting b = 3')
	b = 3;
end

% Python Matlab engine convert integers to int64, we need to convert
% back to double
k = double(k);
b = double(b);

tri = Shilush(k); % 3 by 3 identical blocks, each with the same k randomly 
                  % distributed vertices. Return the Delaunay
                  % triangulation
A = Compute_Matrix(tri.ConnectivityList,k,b); % Compute a block-doubly periodic graph Laplacian 
                             % with random coefficients on the edges, with
                             % the edges defined by tri.
points = tri.Points;
end


function A = Compute_Matrix(tri,k,b)

A = sparse(b*k,b*k);

% For convenience, first create the Laplacian matrix of the 3 by 3 block triangulation. 
% Then use only the middle block to construct A.
B = sparse(9*k,9*k);

% Make a standard log-normal random matrix of size k by 9*k. 
% Only part of it will be used
R = -lognrnd(0,1,k,9*k);

tri = sort(tri,2);
for i = 1:length(tri) % Run over all the triangles
	t = tri(i,:); % For convenience, store triangle i in array t
	B(t(1),t(2)) = R(mod(t(1)-1,k)+1,t(2)-t(1)); % Maintains periodicity in the middle block
	B(t(1),t(3)) = R(mod(t(1)-1,k)+1,t(3)-t(1)); % Maintains periodicity in the middle block
	B(t(2),t(3)) = R(mod(t(2)-1,k)+1,t(3)-t(2)); % Maintains periodicity in the middle block
	% Symmetrize
	B(t(2),t(1)) = B(t(1),t(2));
	B(t(3),t(1)) = B(t(1),t(3));
	B(t(3),t(2)) = B(t(2),t(3));
end

% Plug the rows of B corresponding to the (2,2) block into A in the proper places,
% given that B has 3 by 3 blocks, while A has b by b blocks. This will
% define the (2,2) block of A

A((b+1)*k+1:(b+2)*k,1:3*k) = B((3+1)*k+1:(3+2)*k,1:3*k);
A((b+1)*k+1:(b+2)*k,b*k+1:(b+3)*k) = B((3+1)*k+1:(3+2)*k,3*k+1:(3+3)*k);
A((b+1)*k+1:(b+2)*k,2*b*k+1:(2*b+3)*k) = B((3+1)*k+1:(3+2)*k,2*3*k+1:(2*3+3)*k);

% Now create the rest of the doubly periodic A from its (2,2) block.
for ib = 0:b^2-1 % Run over the blocks, starting from 0 for convenience.
	             % ib = b+1 corresponds to block (2,2).
	A(ib*k+1:ib*k+k,ib*k+1:ib*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+1)*k+1:(b+1)*k+k);
	if (mod(ib,b) == 0) % North block
		A(ib*k+1:ib*k+k,(ib-1+b)*k+1:(ib-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,b*k+1:b*k+k);
	else
		A(ib*k+1:ib*k+k,(ib-1)*k+1:(ib-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,b*k+1:b*k+k);
	end
	if (mod(ib,b) == b-1) % South block 
		A(ib*k+1:ib*k+k,(ib+1-b)*k+1:(ib+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+2)*k+1:(b+2)*k+k);
	else
		A(ib*k+1:ib*k+k,(ib+1)*k+1:(ib+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+2)*k+1:(b+2)*k+k); 
	end
	if (ib < b) % West block 
		A(ib*k+1:ib*k+k,(ib-b+b^2)*k+1:(ib-b+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,k+1:k+k);
	else
		A(ib*k+1:ib*k+k,(ib-b)*k+1:(ib-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,k+1:k+k); 
	end
	if (ib >= b^2-b) % East block
		A(ib*k+1:ib*k+k,(ib+b-b^2)*k+1:(ib+b-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+1)*k+1:(2*b+1)*k+k); 
	else
		A(ib*k+1:ib*k+k,(ib+b)*k+1:(ib+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+1)*k+1:(2*b+1)*k+k); 
	end
	if (ib == 0) % NorthWest block
		A(ib*k+1:ib*k+k,(b^2-1)*k+1:(b^2-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
	elseif (mod(ib,b) == 0)
		A(ib*k+1:ib*k+k,(ib-b-1+b)*k+1:(ib-b-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
	elseif (ib < b)		
		A(ib*k+1:ib*k+k,(ib-b-1+b^2)*k+1:(ib-b-1+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
	else
		A(ib*k+1:ib*k+k,(ib-b-1)*k+1:(ib-b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
	end
	if (ib == b-1) % SouthWest block
		A(ib*k+1:ib*k+k,(b^2-b)*k+1:(b^2-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
	elseif (mod(ib,b) == b-1)
		A(ib*k+1:ib*k+k,(ib-b+1-b)*k+1:(ib-b+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
	elseif (ib < b)		
		A(ib*k+1:ib*k+k,(ib-b+1+b^2)*k+1:(ib-b+1+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
	else
		A(ib*k+1:ib*k+k,(ib-b+1)*k+1:(ib-b+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
	end
	if (ib == b^2-b) % NorthEast block
		A(ib*k+1:ib*k+k,(b-1)*k+1:(b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
	elseif (mod(ib,b) == 0)
		A(ib*k+1:ib*k+k,(ib+b-1+b)*k+1:(ib+b-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
	elseif (ib >= b^2-b)		
		A(ib*k+1:ib*k+k,(ib+b-1-b^2)*k+1:(ib+b-1-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
	else
		A(ib*k+1:ib*k+k,(ib+b-1)*k+1:(ib+b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
	end	
	if (ib == b^2-1) % SouthEast block
		A(ib*k+1:ib*k+k,1:k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
	elseif (mod(ib,b) == b-1)
		A(ib*k+1:ib*k+k,(ib+b+1-b)*k+1:(ib+b+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
	elseif (ib >= b^2-b)		
		A(ib*k+1:ib*k+k,(ib+b+1-b^2)*k+1:(ib+b+1-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
	else
		A(ib*k+1:ib*k+k,(ib+b+1)*k+1:(ib+b+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
	end	

end

% Zerosum
for i = 1:length(A)
	A(i,i) = -sum(A(i,:));
end

% Python Matlab engine does not support sparse arrays
A = full(A);
end

function tri = Shilush(k)

x = rand(k,1);
y = rand(k,1);
X = [x-1;x-1;x-1;x;x;x;x+1;x+1;x+1];
Y = [y-1;y;y+1;y-1;y;y+1;y-1;y;y+1];
tri = delaunayTriangulation(X,Y);

end
