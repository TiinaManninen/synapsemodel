% Plots of inputs, saved state variables, and other saved ouput variables with certain temporal difference
% Tiina Manninen
% September 2020 

clear all;

% Loading data
folderName1 = 'results_post_pre_pairing_100x';
folderName2 = '10ms';
statevar = load(fullfile('..', folderName1, folderName2, 'state_var_results.mat'));
othervar = load(fullfile('..', folderName1, folderName2, 'other_var_results.mat'));
timestim = load(fullfile('..', folderName1, folderName2, 'time_stimuli.mat'));
par = load(fullfile('..', folderName1, folderName2, 'stimulation_parameters.mat'));


% Plotting inputs
figure;
set(gcf,'Name', 'Pre- and postsynaptic stimulus');
subplot(2,1,1);
plot(timestim.time, timestim.I_ext_pre(1:length(timestim.time)),'k');
xlabel('Time (s)');
ylabel('{I_{extpre}}','FontSize', 11,'FontName', 'times');
title({['Temporal difference: -', num2str(par.T_shift), ' ms'] ['Presynaptic stimulus: ', num2str(par.pulserate), ' Hz']});

subplot(2,1,2);
plot(timestim.time, timestim.I_ext_post,'k');
xlabel('Time (s)');
ylabel('{I_{extpost}}','FontSize', 11,'FontName', 'times');
title(['Postsynaptic stimulus: ', num2str(par.pulserate), ' Hz']);

% Plotting saved state variables
figure;
hold all;
set(gcf,'Name', 'State variables');
statenames = fieldnames(statevar);
for i = 1:1:length(statenames)
	subplot(4,7,i);
	hold all;
	x = statevar.(statenames{i});
	plot(timestim.time, x);
	xlabel('Time (s)'); ylabel(char(statenames(i)),'Interpreter','none');
end
hold off;

%% Plotting other saved output variables
figure;
hold all;
set(gcf,'Name', 'Other output variables');
othernames = fieldnames(othervar);
for i = 1:1:length(othernames)
	subplot(3,5,i);
	hold all;
	x = othervar.(othernames{i});
	plot(timestim.time(2:end), x);
	xlabel('Time (s)'); ylabel(char(othernames(i)),'Interpreter','none');
end
hold off;




