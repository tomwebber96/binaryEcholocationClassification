% Specify the directory containing the .pgdf files
directory = 'F:\Latheron\binary'; % Change this to your directory

% Get a list of all .pgdf files in the directory and subdirectories
pgdfFiles = dir(fullfile(directory, '**', '*Clicks*.pgdf'));

% Loop through each .pgdf file
for k = 1:length(pgdfFiles)
    % Get the full path to the .pgdf file
    pgdfFile = fullfile(pgdfFiles(k).folder, pgdfFiles(k).name);
    
    % Load the binary data from the .pgdf file
    [binarydata, fileinfo] = loadPamguardBinaryFile(pgdfFile);
    
    % Create a .mat file name based on the .pgdf file name
    [~, name, ~] = fileparts(pgdfFiles(k).name);
    matFileName = fullfile(pgdfFiles(k).folder, [name, '.mat']);
    
    % Save the binarydata structure as a .mat file in the same directory as the .pgdf file
    save(matFileName, 'binarydata');
end