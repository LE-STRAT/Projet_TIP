%% TIP Food-11 : modèle final (meilleure configuration du banc d'essai)
% Configuration gagnante du banc d'essai (run GPU du 07/11, essai "res18_224_lb_aug") :
%   F1 macro = 0,907 | frugalité = 0,631 | environ 10 min sur GPU
%   - réseau : ResNet-18 pré-entraîné (ImageNet), dernière couche remplacée (11 classes)
%   - images : 224x224, letterbox (image entière, sans déformation, bandes noires)
%   - augmentation : miroir horizontal + rotation aléatoire de -15° à +15° (entraînement seulement)
%   - Adam, learning rate 3e-4 divisé par 10 à mi-parcours, 10 epochs, batch de 64
% Prérequis : Deep Learning Toolbox + "Deep Learning Toolbox Model for ResNet-18 Network".
% GPU fortement conseillé (Parallel Computing Toolbox + GPU NVIDIA) : sur CPU, compter plusieurs heures.
% Utilisation : Run (F5). Le script entraîne, évalue, puis écrit un JSON unique dans le dossier predictions/.
clear; clc; close all;

%% 0. Réglages (ceux de l'essai gagnant : ne pas modifier pour reproduire le résultat)
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");

netName   = "resnet18";
inputSize = 224;
maxEpochs = 10;
batchSize = 64;
learnRate = 3e-4;

if canUseGPU
    fprintf("GPU détecté : entraînement sur GPU (environ 10 min).\n");
else
    warning("Aucun GPU détecté : entraînement sur CPU, compter plusieurs heures.");
end

%% 1. Données : 80 % entraînement / 20 % validation (même découpage que le banc d'essai)
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("%d images d'entraînement, %d de validation, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

% Letterbox + augmentation à l'entraînement ; letterbox seul pour la validation et le test
trainDS = combine(transform(imdsTrain, @(x) {padResize(x, inputSize, true)}),  arrayDatastore(imdsTrain.Labels));
valDS   = combine(transform(imdsVal,   @(x) {padResize(x, inputSize, false)}), arrayDatastore(imdsVal.Labels));

%% 2. Réseau pré-entraîné, dernière couche remplacée (11 classes)
net = imagePretrainedNetwork(netName, NumClasses=numClasses);
nParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf("Paramètres : %.2f M\n", nParams/1e6);

%% 3. Entraînement
itersPerEpoch = max(1, floor(numel(imdsTrain.Files)/batchSize));
opts = trainingOptions("adam", ...
    InitialLearnRate=learnRate, MiniBatchSize=batchSize, MaxEpochs=maxEpochs, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=floor(maxEpochs/2), ...
    Shuffle="every-epoch", ValidationData=valDS, ValidationFrequency=itersPerEpoch, ...
    Metrics="accuracy", Plots="training-progress", Verbose=true, VerboseFrequency=100, ...
    ExecutionEnvironment="auto");
tStart = tic;
net = trainnet(trainDS, net, "crossentropy", opts);
fprintf("Entraînement terminé en %.1f min\n", toc(tStart)/60);

%% 4. Évaluation sur la validation : F1 macro, exactitude, frugalité
valImgs = transform(imdsVal, @(x) {padResize(x, inputSize, false)});
YPred = scores2label(minibatchpredict(net, valImgs), classNames);
YTrue = imdsVal.Labels;
C    = confusionmat(YTrue, YPred);
prec = diag(C) ./ sum(C,1)';
rec  = diag(C) ./ sum(C,2);
f1   = 2*prec.*rec ./ (prec+rec);
F1   = mean(f1, "omitnan");
R    = nParams/1e6;
frug = F1 - 0.3/(1 + 1/R);
fprintf("F1 macro = %.4f | Exactitude = %.2f %% | Frugalité = %.4f\n", ...
    F1, 100*mean(YPred == YTrue), frug);
figure(Name="Matrice de confusion (validation)"); confusionchart(YTrue, YPred);

% Sauvegarde du modèle entraîné
save(fullfile(baseDir, "modele_final.mat"), "net", "classNames", "inputSize", "-v7.3");

%% 5. Prédictions sur le test -> fichier JSON unique dans predictions/ (format {"nom_image": "classe"})
imdsTest = imageDatastore(testDir);
testImgs = transform(imdsTest, @(x) {padResize(x, inputSize, false)});
YTest = scores2label(minibatchpredict(net, testImgs), classNames);

savePredictions(baseDir, "final_res18_224_lb_aug", F1, frug, imdsTest.Files, YTest);

%% Fonctions locales
function out = padResize(img, sz, doAug)
% Letterbox : réduit l'image en gardant ses proportions, puis la centre sur un carré noir.
% doAug = true : miroir horizontal (1 fois sur 2) et rotation aléatoire de -15° à +15°.
if size(img,3) == 1, img = repmat(img, [1 1 3]); end
if size(img,3) > 3, img = img(:,:,1:3); end
if doAug
    if rand > 0.5, img = fliplr(img); end
    img = imrotate(img, -15 + 30*rand, "bilinear", "crop");
end
img = imresize(img, sz / max(size(img,1:2)));
img = img(1:min(end,sz), 1:min(end,sz), :);
out = zeros(sz, sz, 3, "single");
r  = floor((sz - size(img,1))/2) + 1;
cc = floor((sz - size(img,2))/2) + 1;
out(r:r+size(img,1)-1, cc:cc+size(img,2)-1, :) = single(img);
end

function f = savePredictions(baseDir, tag, F1, frug, files, labels)
% Écrit les prédictions du test dans un fichier JSON UNIQUE (jamais écrasé),
% dans le dossier "predictions" : pred_<date_heure>_<configuration>_F1-<F1>_frug-<frugalité>.json
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
