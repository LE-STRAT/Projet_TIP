%% Food-11 : modèle final (ResNet-18 pré-entraîné, réentraînement complet)
% Images 224x224 en letterbox, augmentation (miroir, rotation de -15 à +15 degrés),
% Adam avec un learning rate de 3e-4 divisé par 10 à mi-parcours, 10 epochs, batch de 64.
% Nécessite la Deep Learning Toolbox, le package ResNet-18 et l'Image Processing Toolbox. GPU conseillé.
clear; clc; close all;

%% 0. Réglages
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

%% 1. Données : 80 % entraînement / 20 % validation, même découpage que le banc d'essai
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("%d images d'entraînement, %d de validation, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

% Augmentation sur l'entraînement uniquement
trainDS = combine(transform(imdsTrain, @(x) {padResize(x, inputSize, true)}),  arrayDatastore(imdsTrain.Labels));
valDS   = combine(transform(imdsVal,   @(x) {padResize(x, inputSize, false)}), arrayDatastore(imdsVal.Labels));

%% 2. Réseau pré-entraîné, dernière couche remplacée par 11 sorties
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
[net, info] = trainnet(trainDS, net, "crossentropy", opts);
fprintf("Entraînement terminé en %.1f min\n", toc(tStart)/60);

%% 4. Évaluation sur la validation
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

% Figures 10a et 10b du rapport
figDir = fullfile(baseDir, "figures_rapport");
saveCurves(info, fullfile(figDir, "Fig10a_courbes_final.png"));
saveConfusion(YTrue, YPred, fullfile(figDir, "Fig10b_confusion_final.png"));

save(fullfile(baseDir, "modele_final.mat"), "net", "classNames", "inputSize", "-v7.3");

%% 5. Prédictions sur le test, au format {"nom_image": "classe"}
imdsTest = imageDatastore(testDir);
testImgs = transform(imdsTest, @(x) {padResize(x, inputSize, false)});
YTest = scores2label(minibatchpredict(net, testImgs), classNames);

savePredictions(baseDir, "final_res18_224_lb_aug", F1, frug, imdsTest.Files, YTest);

%% Fonctions locales
function out = padResize(img, sz, doAug)
% Letterbox : réduit l'image sans la déformer et la centre sur un carré noir.
% doAug : miroir horizontal une fois sur deux et rotation aléatoire de -15 à +15 degrés.
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

function saveCurves(info, fichier)
% Courbes de perte et d'exactitude (entraînement et validation), au format des figures du rapport.
try
    if ~isfolder(fileparts(fichier)), mkdir(fileparts(fichier)); end
    T = info.TrainingHistory;
    V = info.ValidationHistory;  V = V(V.Iteration > 0, :);
    fig = figure(Color="w", Units="centimeters", Position=[2 2 16 11]);
    tiledlayout(fig, 2, 1, TileSpacing="compact", Padding="compact");
    motifs = ["loss" "accuracy"];  titres = ["Perte" "Exactitude (%)"];
    for k = 1:2
        ax = nexttile;  hold(ax, "on");  grid(ax, "on");
        cT = find(contains(lower(string(T.Properties.VariableNames)), motifs(k)), 1);
        cV = find(contains(lower(string(V.Properties.VariableNames)), motifs(k)), 1);
        if ~isempty(cT), plot(ax, T.Iteration, smoothdata(T{:, cT}, "movmean", 25), Color=[0 0.447 0.741], LineWidth=1.2); end
        if ~isempty(cV), plot(ax, V.Iteration, V{:, cV}, "-o", Color=[0.851 0.325 0.098], LineWidth=1.2, MarkerSize=4); end
        ylabel(ax, titres(k));  set(ax, FontName="Times New Roman", FontSize=11);
    end
    xlabel(ax, "Itération");
    legend(ax, ["Entraînement" "Validation"], Location="best");
    exportgraphics(fig, fichier, Resolution=300);
catch err
    warning("Courbes non enregistrées : %s", err.message);
end
end

function saveConfusion(yVrai, yPredit, fichier)
% Matrice de confusion avec le rappel par classe, au format des figures du rapport.
try
    if ~isfolder(fileparts(fichier)), mkdir(fileparts(fichier)); end
    fig = figure(Color="w", Units="centimeters", Position=[2 2 16 12]);
    cm = confusionchart(fig, yVrai, yPredit, RowSummary="row-normalized");
    cm.Title = "";  cm.XLabel = "Classe prédite";  cm.YLabel = "Classe réelle";
    cm.FontName = "Times New Roman";  cm.FontSize = 10;
    exportgraphics(fig, fichier, Resolution=300);
catch err
    warning("Matrice de confusion non enregistrée : %s", err.message);
end
end
