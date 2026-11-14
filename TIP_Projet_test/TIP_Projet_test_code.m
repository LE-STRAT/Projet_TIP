%% Food-11 : script de départ (ResNet-18 pré-entraîné, 128 px), référence de la partie 2
% Nécessite la Deep Learning Toolbox et le support package du réseau choisi.
clear; clc;

%% 1. Données
baseDir  = fileparts(mfilename('fullpath')); if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");

imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
classNames = categories(imds.Labels);
numClasses = numel(classNames);

%% 2. Réseau pré-entraîné : on remplace la dernière couche (11 classes)
netName = "resnet18";
net = imagePretrainedNetwork(netName, NumClasses=numClasses);
inputSize = [128 128 3];

% Nombre de paramètres, pour le score de frugalité
nParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf("Paramètres : %.2f M\n", nParams/1e6);

% Redimensionnement seul, sans augmentation
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ColorPreprocessing="gray2rgb");
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal, ColorPreprocessing="gray2rgb");

%% 3. Entraînement
opts = trainingOptions("adam", ...
    InitialLearnRate=1e-3, MiniBatchSize=64, MaxEpochs=5, ...
    Shuffle="every-epoch", ValidationData=augVal, ValidationFrequency=300, ...
    Metrics="accuracy", Plots="training-progress", Verbose=false);
net = trainnet(augTrain, net, "crossentropy", opts);

%% 4. Évaluation sur la validation : F1 macro
scores = minibatchpredict(net, augVal);
YPred  = scores2label(scores, classNames);
YTrue  = imdsVal.Labels;
C = confusionmat(YTrue, YPred);
prec = diag(C) ./ sum(C,1)';
rec  = diag(C) ./ sum(C,2);
f1   = 2*prec.*rec ./ (prec+rec);
F1   = mean(f1, "omitnan");
R    = nParams/1e6;
frug = F1 - 0.3/(1 + 1/R);
fprintf("F1 macro = %.4f | Frugalité = %.4f\n", F1, frug);
figure; confusionchart(YTrue, YPred);

%% 5. Prédictions sur le test
imdsTest = imageDatastore(testDir);
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest, ColorPreprocessing="gray2rgb");
YTest    = scores2label(minibatchpredict(net, augTest), classNames);

savePredictions(baseDir, sprintf("base_%s_%dpx", netName, inputSize(1)), F1, frug, imdsTest.Files, YTest);

%% Fonction locale
function f = savePredictions(baseDir, tag, F1, frug, files, labels)
% Écrit les prédictions dans predictions/, sous un nom unique (date, configuration, scores).
if isempty(baseDir), baseDir = pwd; end
outDir = fullfile(baseDir, "predictions");
if ~isfolder(outDir), mkdir(outDir); end
stamp = string(datetime("now", Format="yyyyMMdd_HHmmss"));
f = fullfile(outDir, sprintf("pred_%s_%s_F1-%.4f_frug-%.4f.json", stamp, tag, F1, frug));
fid = fopen(f, "w");
fprintf(fid, "{\n");
n = numel(files);
for i = 1:n
    [~, name] = fileparts(files{i});
    sep = ","; if i == n, sep = ""; end
    fprintf(fid, '    "%s": "%s"%s\n', name, lower(string(labels(i))), sep);
end
fprintf(fid, "}\n");
fclose(fid);
fprintf("Prédictions écrites : %s\n", f);
end
