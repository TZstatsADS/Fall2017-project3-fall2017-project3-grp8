addpath(genpath('.'));
datasets = {'Real','Fake'};   
dinfo = dir('new_images/*.jpg'); % make sure it is the clean data directory
names_cell = {dinfo.name};      % get image names in the directory
train_lists = {{names_cell{1}},{names_cell{2}}};    % specify lists of train images   
test_lists = {{names_cell{1:3000}},{names_cell{1}}};     % specify lists of test images
feature = 'gist';                                               % specify feature to use 
c = conf();                                                       % load the config structure
datasets_feature(datasets, train_lists, test_lists, feature, c);  % perform feature extraction
%train_features = load_feature(datasets{1}, feature, 'train', c);  % load train features
test_features = load_feature(datasets{1}, feature, 'test', c);    % load test features
csvwrite('features_GIST.csv',test_features );
