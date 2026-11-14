%% Food-11 : CNN léger entraîné sans pré-entraînement (Matlab R2024a+)
% Frugalité = F1 - 0.3 * R/(1+R), avec R le nombre de paramètres en millions.
clear; clc; close all;
rng(42);

%% 1. Paramètres
dossierTrain = "food-11/train";
dossierTest  = "food-11/test";
tailleImg    = [128 128 3];
canaux       = [24 48 96 160 256 256];      % filtres par bloc : fixe la taille du réseau
nbEpoques    = 60;
tailleBatch  = 64;
lrInitial    = 2e-3;

%% 2. Données
imds = imageDatastore(dossierTrain, ...
    IncludeSubfolders=true, LabelSource="foldernames");

classes    = categories(imds.Labels);
nbClasses  = numel(classes);
fprintf("Classes détectées (%d) : %s\n", nbClasses, strjoin(string(classes), ", "));

% 80 % entraînement / 20 % validation, classe par classe
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("Train : %d images | Validation : %d images\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files));

% Augmentation, sur l'entraînement uniquement
augmenteur = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-20 20], ...
    RandScale=[0.85 1.15], ...
    RandXTranslation=[-8 8], ...
    RandYTranslation=[-8 8]);

augTrain = augmentedImageDatastore(tailleImg(1:2), imdsTrain, ...
    DataAugmentation=augmenteur, ColorPreprocessing="gray2rgb");
augVal   = augmentedImageDatastore(tailleImg(1:2), imdsVal, ...
    ColorPreprocessing="gray2rgb");

%% 3. Architecture
c = canaux;
couches = [
    imageInputLayer(tailleImg, Normalization="none", Name="entree")
    % pas de normalisation d'entrée : la BatchNorm qui suit la 1re convolution s'en charge

    % Stem
    convolution2dLayer(3, c(1), Padding="same", Name="stem_conv")
    batchNormalizationLayer(Name="stem_bn")
    reluLayer(Name="stem_relu")
    maxPooling2dLayer(2, Stride=2, Name="stem_pool")            % 128 -> 64

    % Bloc 1 : convolution classique
    convolution2dLayer(3, c(2), Padding="same", Name="b1_conv")
    batchNormalizationLayer(Name="b1_bn")
    reluLayer(Name="b1_relu")
    maxPooling2dLayer(2, Stride=2, Name="b1_pool")              % 64 -> 32

    % Blocs 2 à 5 : convolutions séparables
    blocSeparable(c(3), "b2")
    maxPooling2dLayer(2, Stride=2, Name="b2_pool")              % 32 -> 16
    blocSeparable(c(4), "b3")
    maxPooling2dLayer(2, Stride=2, Name="b3_pool")              % 16 -> 8
    blocSeparable(c(5), "b4")
    maxPooling2dLayer(2, Stride=2, Name="b4_pool")              % 8 -> 4
    blocSeparable(c(6), "b5")

    % Classification
    globalAveragePooling2dLayer(Name="gap")
    dropoutLayer(0.3, Name="dropout")
    fullyConnectedLayer(nbClasses, Name="fc")
    softmaxLayer(Name="softmax")
];

net = dlnetwork(couches);

% Nombre de paramètres apprenables
nbParams = sum(cellfun(@numel, net.Learnables.Value));
R = nbParams / 1e6;
fprintf("\nNombre de paramètres : %d (R = %.4f M)\n", nbParams, R);
fprintf("Pénalité de frugalité : %.4f\n", 0.3 * (R / (1 + R)));
if nbParams < 5e4 || nbParams > 5e5
    warning("Nombre de paramètres hors de la plage visée (50k - 500k).");
end

%% 4. Entraînement
nbIterParEpoque = floor(numel(imdsTrain.Files) / tailleBatch);

options = trainingOptions("adam", ...
    InitialLearnRate=lrInitial, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.3, ...
    LearnRateDropPeriod=10, ...
    L2Regularization=1e-4, ...
    MaxEpochs=nbEpoques, ...
    MiniBatchSize=tailleBatch, ...
    Shuffle="every-epoch", ...
    ValidationData=augVal, ...
    ValidationFrequency=nbIterParEpoque, ...
    OutputNetwork="best-validation", ...     % réseau de la meilleure epoch en validation
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ExecutionEnvironment="auto");

[net, info] = trainnet(augTrain, net, "crossentropy", options);

save("cnn_frugal_food11.mat", "net", "classes", "tailleImg");

%% 5. Évaluation sur la validation
% Mêmes images que imdsVal, sans les labels, pour la prédiction
augValPred = augmentedImageDatastore(tailleImg(1:2), ...
    imageDatastore(imdsVal.Files), ColorPreprocessing="gray2rgb");

scoresVal = minibatchpredict(net, augValPred, MiniBatchSize=128);
predVal   = scores2label(scoresVal, classes);

% Matrice de confusion : lignes = classe réelle, colonnes = classe prédite
yVrai = double(imdsVal.Labels);
yPred = double(categorical(predVal, classes));
C = accumarray([yVrai yPred], 1, [nbClasses nbClasses]);

vp        = diag(C);
precision = vp ./ sum(C, 1)';
rappel    = vp ./ sum(C, 2);
f1Classe  = 2 * precision .* rappel ./ (precision + rappel);
f1Classe(isnan(f1Classe)) = 0;                              % classe jamais prédite
f1Macro   = mean(f1Classe);

frug = f1Macro - 0.3 * (R / (1 + R));

fprintf("\n================ RÉSULTATS (validation) ================\n");
fprintf("Accuracy        : %.4f\n", sum(vp) / sum(C, "all"));
fprintf("F1 macro        : %.4f\n", f1Macro);
fprintf("Paramètres      : %d (R = %.4f M)\n", nbParams, R);
fprintf("Score frugalité : %.4f\n", frug);
fprintf("=========================================================\n");
disp(table(classes, round(f1Classe, 3), VariableNames=["Classe" "F1"]));

% Figures 4a et 4b du rapport
figDir = fullfile(fileparts(mfilename('fullpath')), "figures_rapport");
saveCurves(info, fullfile(figDir, "Fig4a_courbes_sans_transfert.png"));
saveConfusion(imdsVal.Labels, predVal, fullfile(figDir, "Fig4b_confusion_sans_transfert.png"));

%% 6. Prédictions sur le test
imdsTest = imageDatastore(dossierTest);
augTest  = augmentedImageDatastore(tailleImg(1:2), imdsTest, ...
    ColorPreprocessing="gray2rgb");

scoresTest = minibatchpredict(net, augTest, MiniBatchSize=128);
predTest   = scores2label(scoresTest, classes);

[~, nomsImages] = cellfun(@fileparts, imdsTest.Files, UniformOutput=false);
nomsClasses     = cellstr(lower(string(predTest)));

% containers.Map et non struct : les noms d'images ("0", "1"...) ne sont pas des noms de champs valides
dico = containers.Map(nomsImages, nomsClasses);
txtJson = jsonencode(dico, PrettyPrint=true);

predFile = predictionsPath(fileparts(mfilename('fullpath')), ...
    sprintf("sansTransfert_%dpx", tailleImg(1)), f1Macro, frug);
fid = fopen(predFile, "w");
fwrite(fid, txtJson, "char");
fclose(fid);
fprintf("\nPrédictions écrites (%d images) : %s\n", numel(nomsImages), predFile);

%% Fonctions locales
function couches = blocSeparable(nbFiltres, nom)
% Convolution séparable : 3x3 par canal (depthwise) puis 1x1 (pointwise),
% soit 9*Cin + Cin*Cout paramètres au lieu de 9*Cin*Cout.
    couches = [
        groupedConvolution2dLayer(3, 1, "channel-wise", Padding="same", Name=nom+"_dw")
        batchNormalizationLayer(Name=nom+"_dw_bn")
        reluLayer(Name=nom+"_dw_relu")
        convolution2dLayer(1, nbFiltres, Name=nom+"_pw")
        batchNormalizationLayer(Name=nom+"_pw_bn")
        reluLayer(Name=nom+"_pw_relu")
    ];
end

function f = predictionsPath(baseDir, tag, F1, frug)
% Nom de fichier unique dans predictions/ (date, configuration, scores).
if isempty(baseDir), baseDir = pwd; end
outDir = fullfile(baseDir, "predictions");
if ~isfolder(outDir), mkdir(outDir); end
stamp = string(datetime("now", Format="yyyyMMdd_HHmmss"));
f = fullfile(outDir, sprintf("pred_%s_%s_F1-%.4f_frug-%.4f.json", stamp, tag, F1, frug));
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
