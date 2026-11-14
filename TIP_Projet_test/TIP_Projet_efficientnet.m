%% Food-11 : meilleur modèle en frugalité (EfficientNet-b0 pré-entraîné, réentraînement complet)
% Reprend la configuration de l'essai "E_effb0" du banc d'essai (F1 = 0,925, frugalité = 0,685 en validation),
% pour vérifier que le résultat se reproduit.
% Nécessite la Deep Learning Toolbox et le package "Deep Learning Toolbox Model for EfficientNet-b0 Network".
% GPU conseillé : l'essai d'origine a duré plus de 5 h.
clear; clc; close all;

%% 0. Réglages
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");

netName   = "efficientnetb0";
inputSize = 224;
maxEpochs = 15;        % à aligner sur la colonne "epochs" de E_effb0 dans resultats.csv
batchSize = 64;
learnRate = 3e-4;
graine    = 0;         % graine de l'essai d'origine

if canUseGPU
    fprintf("GPU détecté.\n");
else
    warning("Aucun GPU détecté : l'entraînement sera très long sur CPU.");
end

%% 1. Données : 80 % entraînement / 20 % validation, même découpage que le banc d'essai
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("%d images d'entraînement, %d de validation, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

% Images étirées en 224x224 (pas de letterbox). Augmentation sur l'entraînement uniquement.
aug = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15]);
trainDS = augmentedImageDatastore([inputSize inputSize], imdsTrain, ColorPreprocessing="gray2rgb", DataAugmentation=aug);
valDS   = augmentedImageDatastore([inputSize inputSize], imdsVal, ColorPreprocessing="gray2rgb");

%% 2. Réseau pré-entraîné, dernière couche remplacée par 11 sorties
rng(graine);
net = imagePretrainedNetwork(netName, NumClasses=numClasses);
nParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf("Paramètres : %.2f M\n", nParams/1e6);

%% 3. Entraînement
itersPerEpoch = max(1, floor(numel(imdsTrain.Files)/batchSize));
opts = trainingOptions("adam", ...
    InitialLearnRate=learnRate, MiniBatchSize=batchSize, MaxEpochs=maxEpochs, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=max(1, floor(maxEpochs/2)), ...
    Shuffle="every-epoch", ValidationData=valDS, ValidationFrequency=itersPerEpoch, ...
    OutputNetwork="best-validation", ...
    Metrics="accuracy", Plots="training-progress", Verbose=true, VerboseFrequency=100, ...
    ExecutionEnvironment="auto");
tStart = tic;
[net, info] = trainnet(trainDS, net, "crossentropy", opts);
fprintf("Entraînement terminé en %.1f min\n", toc(tStart)/60);

%% 4. Évaluation sur la validation
valImgs = augmentedImageDatastore([inputSize inputSize], imdsVal, ColorPreprocessing="gray2rgb");
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
disp(table(string(classNames), round(f1, 3), round(rec, 3), VariableNames=["Classe" "F1" "Rappel"]));

% Figures 10a et 10b du rapport
figDir = fullfile(baseDir, "figures_rapport");
saveCurves(info, fullfile(figDir, "Fig10a_courbes_efficientnet.png"));
saveConfusion(YTrue, YPred, fullfile(figDir, "Fig10b_confusion_efficientnet.png"));

%% 5. Prédictions sur le test, au format {"nom_image": "classe"}
imdsTest = imageDatastore(testDir);
testImgs = augmentedImageDatastore([inputSize inputSize], imdsTest, ColorPreprocessing="gray2rgb");
YTest = scores2label(minibatchpredict(net, testImgs), classNames);
savePredictions(baseDir, "efficientnetb0_224_aug", F1, frug, imdsTest.Files, YTest);

%% Fonctions locales
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
