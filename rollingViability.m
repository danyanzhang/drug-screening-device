function cells2 = rollingViability(cells, r)
% Rolling ball method for viability based on cells within radius r

cells2 = cells;
x = cells.x;
y = cells.y;
viability = x;
live = cells.live;

% For each x,y cell, find indices of closest x,y within r
Idx = rangesearch([x,y],[x,y],r);

for i = 1:length(Idx) % for each cell
	% average live status of neighbors
	viability(i) = mean(live(Idx{i}));
end

cells2.viability = viability;
end