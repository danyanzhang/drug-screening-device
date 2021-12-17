% Creates "smile plots" for evaluation of pairwise drug synergy
% from continuous screen data from microfluidic device
% Author: Dan Zhang
% Updated: 2021-12-16


clear;clc;close all
addpath(['..', filesep, 'DataDerived'])

condition = '3drug'; % options are 2drug, 3drug, and control
% from microfluidic device experiments, each dev is a replicate

% Drug ordering
% Drug 1: Methotrexate
% Drug 2: MRX-2843
% Drug 3: Vincristine

% colors for drug pairs (RGB, normalized to matlab [0 1] scaling)
% Ordering of pairs: 2-3, 1-3, 1-2
color.darkBlue = [0, 63, 92]/255;
color.magenta = [188, 80, 144]/255;
color.orange = [255, 166, 0]/255;


switch condition % load the appropriate datasets
case '2drug'
    T1 = readtable('20210920_dev4.csv');
    T2 = readtable('20210920_dev5.csv');
    T3 = readtable('20210920_dev6.csv');
case '3drug'
    T1 = readtable('20210920_dev1.csv');
    T2 = readtable('20210920_dev2.csv');
    T3 = readtable('20210920_dev3.csv');
case 'control'
    T1 = readtable('20210920_dev7.csv');
    T2 = readtable('20210920_dev8.csv');
    T3 = readtable('20210920_dev9.csv');
end

T = [T1; T2; T3]; % stack replicates into 1 table

% rename variables to match cells format
T = renamevars(T,["x","y","C1","C2","C3"], ...
                 ["X","Y","c1","c2","c3"]);

% load ternary plot packages
addpath('TernaryPlot')
addpath('ternary2')

% create zones
cells = T;
thresholdMainZone = 0.08; % below 8% of drug is ignored
cells.zone = zeros(height(cells), 1);
cells.viability = zeros(height(cells), 1);
[cells.ternX, cells.ternY] = ternCoord(cells.c1, cells.c2, cells.c3);

% zone 1: interaction between drug 1 and 2 (no 3)
% get threshold based on max
idx{1} = find(cells.c1 < thresholdMainZone);
idx{2} = find(cells.c2 < thresholdMainZone);
idx{3} = find(cells.c3 < thresholdMainZone);

% assign zone number in table
for i = 1:length(idx)
    cells.zone(idx{i}) = i;
end
% zone 0: middle portion of device
% zone 1: interaction between 2 and 3
% zone 2: interaction between 1 and 3
% zone 3: interaction between 1 and 2


%------------------------------------------------------------------------------
% Figure 1. Pairwise lines plotted against ratio B/total
figure
hold on
filter_width = 0.1;
[T23, T13, T12] = isolatePair(T, 8); % 8% of drug
[x23, viability23] = viabilityOverDrugRatio(T23, filter_width);
[x13, viability13] = viabilityOverDrugRatio(T13, filter_width);
[x12, viability12] = viabilityOverDrugRatio(T12, filter_width);

plot(x23, medfilt1(smooth(viability23, .2)), '-', 'LineWidth', 3, 'Color', color.darkBlue)
plot(x13, medfilt1(smooth(viability13, .2)), '-', 'LineWidth', 3, 'Color', color.magenta)
plot(x12, medfilt1(smooth(viability12, .2)), '-', 'LineWidth', 3, 'Color', color.orange)

legend({"MRX-VCR", "MTX-VCR", "MTX-MRX"}, 'Location', 'SouthEast')
xlabel('Ratio Drug B/A')
ylabel('Viability (%)')
set(gca,'FontSize',14)


%% AUC of T1, T2, and T3 separately
T1 = renamevars(T1,["x","y","C1","C2","C3"], ...
    ["X","Y","c1","c2","c3"]);
T2 = renamevars(T2,["x","y","C1","C2","C3"], ...
    ["X","Y","c1","c2","c3"]);
T3 = renamevars(T3,["x","y","C1","C2","C3"], ...
    ["X","Y","c1","c2","c3"]);            

[T23_1, T13_1, T12_1] = isolatePair(T1, 8); % different replicates
[T23_2, T13_2, T12_2] = isolatePair(T2, 8);
[T23_3, T13_3, T12_3] = isolatePair(T3, 8);



%------------------------------------------------------------------------------
% Figure 2. Viability bar chart, replicates
meanViability23(1) = avgViability(T23_1);
meanViability13(1) = avgViability(T13_1);
meanViability12(1) = avgViability(T12_1);

meanViability23(2) = avgViability(T23_2);
meanViability13(2) = avgViability(T13_2);
meanViability12(2) = avgViability(T12_2);

meanViability23(3) = avgViability(T23_3);
meanViability13(3) = avgViability(T13_3);
meanViability12(3) = avgViability(T12_3);

figure
bar_viability_x = categorical({'Replicate 1', 'Replicate 2', 'Replicate 3'});
%bar_viability_x = reordercats(bar_viability_x,{'Viability23', 'Viability13', 'Viability12'});
bar_viability_y = [meanViability23', meanViability13', meanViability12'];

b = bar(bar_viability_x, bar_viability_y);
b(1).FaceColor = color.darkBlue;
b(2).FaceColor = color.magenta;
b(3).FaceColor = color.orange;
ylabel('Average Viability (%)')
legend({"MRX-VCR", "MTX-VCR", "MTX-MRX"}, 'Location', 'bestoutside')
set(gca,'FontSize',14)
ylim([0 1])


%% AUC FROM ADDITIVITY ISOBOLE
AUC23(1) = viabilityAUC(T23_1);
AUC13(1) = viabilityAUC(T13_1);
AUC12(1) = viabilityAUC(T12_1);

AUC23(2) = viabilityAUC(T23_2);
AUC13(2) = viabilityAUC(T13_2);
AUC12(2) = viabilityAUC(T12_2);

AUC23(3) = viabilityAUC(T23_3);
AUC13(3) = viabilityAUC(T13_3);
AUC12(3) = viabilityAUC(T12_3);


%------------------------------------------------------------------------------
% Figure 3. AUC bar chart, replicates

figure
bar_AUC_x = categorical({'Replicate 1', 'Replicate 2', 'Replicate 3'});
%bar_viability_x = reordercats(bar_viability_x,{'Viability23', 'Viability13', 'Viability12'});
bar_AUC_y = [AUC23', AUC13', AUC12'];

b = bar(bar_AUC_x, bar_AUC_y);
b(1).FaceColor = color.darkBlue;
b(2).FaceColor = color.magenta;
b(3).FaceColor = color.orange;
ylabel('AUC from additivity')
legend({"MRX-VCR", "MTX-VCR", "MTX-MRX"}, 'Location', 'bestoutside')
set(gca,'FontSize',14)
ylim([-0.1 0.1])


%------------------------------------------------------------------------------
% Figure 4
[T23, T13, T12] = isolatePair(T, 8); % different replicates
averageViability23 = avgViability(T23);
averageViability13 = avgViability(T13);
averageViability12 = avgViability(T12);

AUC23 = viabilityAUC(T23);
AUC13 = viabilityAUC(T13);
AUC12 = viabilityAUC(T12);

% Aggregated bar charts
figure
bar_viability_x = categorical({'MRX-VCR', 'MTX-VCR', 'MTX-MRX'});
bar_viability_x = reordercats(bar_viability_x,{'MRX-VCR', 'MTX-VCR', 'MTX-MRX'});
bar_viability_y = [averageViability23, averageViability13, averageViability12];

b = bar(bar_viability_x, bar_viability_y, 'facecolor', 'flat');
clr = [color.darkBlue; color.magenta; color.orange];
b.CData = clr;
ylabel('Average Viability (%)')
%legend({"MRX-VCR", "MTX-VCR", "MTX-MRX"}, 'Location', 'bestoutside')
set(gca,'FontSize',14)
ylim([0 1])


figure
bar_AUC_x = categorical({'MRX-VCR', 'MTX-VCR', 'MTX-MRX'});
bar_AUC_x = reordercats(bar_AUC_x,{'MRX-VCR', 'MTX-VCR', 'MTX-MRX'});
bar_AUC_y = [AUC23, AUC13, AUC12];

b = bar(bar_AUC_x, bar_AUC_y, 'facecolor', 'flat');
b.CData = clr;
ylabel('AUC from additivity')
%legend({"MRX-VCR", "MTX-VCR", "MTX-MRX"}, 'Location', 'bestoutside')
set(gca,'FontSize',14)
ylim([-0.1 0.1])

%cells = ternaryZoning(T);
rmpath(['..', filesep, 'DataDerived'])



%% CUSTOM FUNCTIONS ================================================================

% -----------------------------------------------------------------------------
function AUC = viabilityAUC(T)
% Calculate the AUC of the viability data, assuming straight line from left and right sides
% some y=mx+b stuff happening

% create new axis
concentrations = T{:, 3:5};
if mean(concentrations(:,1)) < 0.1
    a = concentrations(:,2);
    b = concentrations(:,3);
elseif mean(concentrations(:,2)) < 0.1
    a = concentrations(:,1);
    b = concentrations(:,3);
elseif mean(concentrations(:,3)) < 0.1
    a = concentrations(:,1);
    b = concentrations(:,2);
end

xaxis = b./(a + b);
[sortedAxis, I] = sort(xaxis);
liveSorted = T.live(I);
filter_width = 0.1;
viability = movmean(liveSorted, height(T).* filter_width);
viability = medfilt1(smooth(viability, .2)); % doubly smoothed line
x = sortedAxis;
y = viability;

% finding the line of additivity
x_line = linspace(x(1), x(end), 20); % creating new x for the straight line
m = (y(end) - y(1))./(x(end) - x(1));
b = y(1) - m.*(x(1));
y_line = m.*x_line + b;

% subtracting AUCs
AUC_additivity = trapz(x_line, y_line);
AUC_data = trapz(x, y);
AUC = AUC_additivity - AUC_data;

% Plotting script
%{
figure
plot(x, y, '-')
hold on
plot(x_line, y_line, '.--')
hold off
axis([0, 1, 0, 1])
%}
end % end of function

% -----------------------------------------------------------------------------
function viability = avgViability(T)
viability = mean(T.live); 
end % end of function

% -----------------------------------------------------------------------------
% isolatePair: extract pairwise data under cutoff
function [T23, T13, T12] = isolatePair(T, cutoffPercent)
% taking a table of cell data, produce only the data with under the percent cutoff (%)
thresholdMainZone = cutoffPercent./100;
cells = T;
cells.zone = zeros(height(cells), 1);
cells.viability = zeros(height(cells), 1);
[cells.ternX cells.ternY] = ternCoord(cells.c1, cells.c2, cells.c3);
% zone 1: interaction between drug 1 and 2 (no 3)
% get threshold based on max
idx{1} = find(cells.c1 < thresholdMainZone);
idx{2} = find(cells.c2 < thresholdMainZone);
idx{3} = find(cells.c3 < thresholdMainZone);

% assign zone number in table
for i = 1:length(idx)
    cells.zone(idx{i}) = i;
end
T23 = cells(idx{1}, :);
T13 = cells(idx{2}, :);
T12 = cells(idx{3}, :);
end % end of function

% -----------------------------------------------------------------------------
% viabilityOverDrugRatio: creates normalized ratiometric axis by viability
function [x, viability] = viabilityOverDrugRatio(T, windowWidth)
% windowWidth defines moving average filter width, max 1

% which drug is missing? 1, 2, or 3
[~, missing] = min([mean(T.c1), mean(T.c2), mean(T.c3)]);
switch missing
case 1
    druga = T.c2;
    drugb = T.c3;
case 2
    druga = T.c1;
    drugb = T.c3;
case 3
    druga = T.c1;
    drugb = T.c2;
end

% create new ratiometric x axis
live = T.live;
sumab = druga + drugb;
xaxis = drugb./sumab;

% sort the axis for averaging
[x, I] = sort(xaxis);
liveSorted = live(I);
viability = movmean(liveSorted, height(T)*windowWidth);
end % end of function

% -----------------------------------------------------------------------------
% ternCoord: convert cartesian to ternary coordinate system
function [ternX, ternY] = ternCoord(a,b,c)
    ternX = 0.5.*(2.*b+c)./(a+b+c);
    ternY = (sqrt(3)./2).*c./(a+b+c);
end % end of function

% -----------------------------------------------------------------------------
function viability = rollingViability(cells, r)
    % Rolling ball method for viability based on cells within radius r
    x = cells.X;
    y = cells.Y;
    live = cells.live;
    viability = cells.X; % initialize
    
    % For each x,y cell, find indices of closest x,y within r
    Idx = rangesearch([x,y],[x,y],r);
    
    for i = 1:length(Idx) % for each cell
        % average live status of neighbors
        viability(i) = mean(live(Idx{i}));
    end    
end % end of function

% -----------------------------------------------------------------------------
function viabilityOriginalSort = viabilityPairTernary(conc1, conc2, live)
% for a linear combination of concentrations of drugs 1 and 2, calculates viability
% as a moving window with increasing ratio of drug 1/2
sum12 = conc1 + conc2;
xaxis = conc2./sum12; % normalize concentration of drug 2
% (so that when sorted, goes from drug 1 to drug 2)

% sort
[sortedAxis, I] = sort(xaxis);
liveSorted = live(I);
viability = movmean(liveSorted, length(conc1).*0.20); % moving average of 20% of cells

[sorted2 sortIndRev] = sort(I);
viabilityOriginalSort = viability(sortIndRev);
end % end of function

% -----------------------------------------------------------------------------
function binnedAUC = ratioSpecificAUC(T)

% create new axis

% sort by new axis


% bin ratios (by 10%) and live dead


% viability from additivity assumption


% ratiometric synergy calculation

% plot bar plot of bins

end % end of function