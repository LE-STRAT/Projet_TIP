%% Food-11 : script complet (CNN sans pré-entraînement, puis EfficientNet-b0 pré-entraîné)
% Entraîne les deux modèles du rapport sur le même découpage, les évalue sur la validation,
% enregistre les figures dans figures_rapport/ et les prédictions du test dans predictions/.
% Nécessite la Deep Learning Toolbox et le package "Deep Learning Toolbox Model for EfficientNet-b0 Network".
clear; clc; close all;

%% 0. Réglages
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");
figDir   = fullfile(baseDir, "figures_rapport");
if ~isfolder(figDir), mkdir(figDir); end

% Modèle 1 : CNN léger, poids aléatoires
cnn = struct(nom="sans_transfert", titre="CNN sans pré-entraînement", figure="Fig4", ...
    taille=128, epochs=60, batch=64, lr=2e-3, dropFactor=0.3, dropPeriod=10, graine=42);
canaux = [24 48 96 160 256 256];          % filtres par bloc : fixe la taille du réseau

% Modèle 2 : EfficientNet-b0 pré-entraîné sur ImageNet, entièrement réentraîné
eff = struct(nom="final", titre="EfficientNet-b0 pré-entraîné", figure="Fig10", ...
    taille=224, epochs=15, batch=64, lr=3e-4, dropFactor=0.1, dropPeriod=7, graine=0);

if ~canUseGPU
    warning("Aucun GPU détecté : les deux entraînements seront très longs sur CPU.");
end

%% 1. Données : 80 % entraînement / 20 % validation, classe par classe
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classes   = categories(imds.Labels);
nbClasses = numel(classes);
rng(0);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
imdsTest = imageDatastore(testDir);
fprintf("%d images d'entraînement, %d de validation, %d de test, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files), nbClasses);

%% 2. Réseaux et augmentation
% Les deux réseaux sont créés avant tout entraînement : un package manquant se voit tout de suite.
rng(cnn.graine);
netCnn = cnnLeger(cnn.taille, canaux, nbClasses);
rng(eff.graine);
netEff = imagePretrainedNetwork("efficientnetb0", NumClasses=nbClasses);

% Augmentation plus forte pour le CNN, qui voit chaque image 60 fois
augCnn = imageDataAugmenter(RandXReflection=true, RandRotation=[-20 20], ...
    RandScale=[0.85 1.15], RandXTranslation=[-8 8], RandYTranslation=[-8 8]);
augEff = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15]);

%% 3. Entraînement et évaluation
resCnn = entrainerEtEvaluer(netCnn, cnn, augCnn, imdsTrain, imdsVal, imdsTest, classes, baseDir, figDir);
resEff = entrainerEtEvaluer(netEff, eff, augEff, imdsTrain, imdsVal, imdsTest, classes, baseDir, figDir);

%% 4. Comparaison des deux modèles
res = [resCnn resEff];
bilan = table([res.titre]', [res.paramsM]', [res.F1]', [res.exactitude]', [res.frugalite]', [res.minutes]', ...
    VariableNames=["Modele" "ParamsM" "F1" "Exactitude" "Frugalite" "Minutes"]);
disp(bilan);
writetable(bilan, fullfile(baseDir, "resultats_complet.csv"));

parClasse = table(string(classes), resCnn.f1Classe, resEff.f1Classe, VariableNames=["Classe" "F1_CNN" "F1_EfficientNet"]);
disp(parClasse);

fig = nouvelleFigure(16, 9);
tiledlayout(fig, 1, 3, TileSpacing="compact", Padding="compact");
ax = nexttile([1 2]);
b = bar(ax, categorical(classes, classes), [resCnn.f1Classe resEff.f1Classe]);
b(1).FaceColor = couleur(1);  b(2).FaceColor = couleur(2);
ylabel(ax, "F1 par classe (validation)");  ylim(ax, [0 1]);
legend(ax, [res.titre], Location="southoutside");
mettreEnForme(ax);
ax = nexttile;
b = bar(ax, categorical(["F1 macro" "Frugalité"], ["F1 macro" "Frugalité"]), [[res.F1]; [res.frugalite]]);
b(1).FaceColor = couleur(1);  b(2).FaceColor = couleur(2);
ylim(ax, [0 1]);
mettreEnForme(ax);
exportgraphics(fig, fullfile(figDir, "Fig11_comparaison.png"), Resolution=300);

%% Fonctions locales
function res = entrainerEtEvaluer(net, cfg, aug, imdsTrain, imdsVal, imdsTest, classes, baseDir, figDir)
% Entraîne un réseau, l'évalue sur la validation, enregistre ses figures et ses prédictions du test.
fprintf("\n=== %s (%d px, %d epochs, LR %g) ===\n", cfg.titre, cfg.taille, cfg.epochs, cfg.lr);
taille = [cfg.taille cfg.taille];
dsTrain = augmentedImageDatastore(taille, imdsTrain, ColorPreprocessing="gray2rgb", DataAugmentation=aug);
dsVal   = augmentedImageDatastore(taille, imdsVal, ColorPreprocessing="gray2rgb");
dsValSansLabels = augmentedImageDatastore(taille, imageDatastore(imdsVal.Files), ColorPreprocessing="gray2rgb");
dsTest  = augmentedImageDatastore(taille, imdsTest, ColorPreprocessing="gray2rgb");

nbParams = sum(cellfun(@numel, net.Learnables.Value));
R = nbParams / 1e6;
fprintf("Paramètres : %d (R = %.3f M, pénalité de frugalité %.3f)\n", nbParams, R, 0.3/(1 + 1/R));

options = trainingOptions("adam", ...
    InitialLearnRate=cfg.lr, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=cfg.dropFactor, LearnRateDropPeriod=cfg.dropPeriod, ...
    L2Regularization=1e-4, ...
    MaxEpochs=cfg.epochs, MiniBatchSize=cfg.batch, Shuffle="every-epoch", ...
    ValidationData=dsVal, ValidationFrequency=max(1, floor(numel(imdsTrain.Files)/cfg.batch)), ...
    OutputNetwork="best-validation", ...          % réseau de la meilleure epoch en validation
    Metrics="accuracy", Plots="training-progress", Verbose=false, ExecutionEnvironment="auto");
chrono = tic;
[net, info] = trainnet(dsTrain, net, "crossentropy", options);
minutes = toc(chrono) / 60;

% Matrice de confusion : lignes = classe réelle, colonnes = classe prédite
predVal = scores2label(minibatchpredict(net, dsValSansLabels, MiniBatchSize=128), classes);
nbClasses = numel(classes);
C = accumarray([double(imdsVal.Labels) double(categorical(predVal, classes))], 1, [nbClasses nbClasses]);
precision = diag(C) ./ sum(C, 1)';
rappel    = diag(C) ./ sum(C, 2);
f1Classe  = 2 * precision .* rappel ./ (precision + rappel);
f1Classe(isnan(f1Classe)) = 0;                    % classe jamais prédite
F1   = mean(f1Classe);
frug = F1 - 0.3/(1 + 1/R);
fprintf("F1 macro = %.4f | Exactitude = %.4f | Frugalité = %.4f | %.1f min\n", ...
    F1, sum(diag(C))/sum(C, "all"), frug, minutes);

tracerCourbes(info, fullfile(figDir, cfg.figure + "a_courbes_" + cfg.nom + ".png"));
tracerConfusion(imdsVal.Labels, predVal, fullfile(figDir, cfg.figure + "b_confusion_" + cfg.nom + ".png"));

predTest = scores2label(minibatchpredict(net, dsTest, MiniBatchSize=128), classes);
ecrirePredictions(baseDir, cfg.nom, F1, frug, imdsTest.Files, predTest);

res = struct(titre=string(cfg.titre), paramsM=R, F1=F1, exactitude=sum(diag(C))/sum(C, "all"), ...
    frugalite=frug, minutes=minutes, f1Classe=f1Classe);
end

function net = cnnLeger(taille, c, nbClasses)
% CNN à 6 blocs : deux convolutions classiques, puis quatre convolutions séparables.
couches = [
    imageInputLayer([taille taille 3], Normalization="none", Name="entree")
    % pas de normalisation d'entrée : la BatchNorm qui suit la 1re convolution s'en charge
    convolution2dLayer(3, c(1), Padding="same", Name="stem_conv")
    batchNormalizationLayer(Name="stem_bn")
    reluLayer(Name="stem_relu")
    maxPooling2dLayer(2, Stride=2, Name="stem_pool")

    convolution2dLayer(3, c(2), Padding="same", Name="b1_conv")
    batchNormalizationLayer(Name="b1_bn")
    reluLayer(Name="b1_relu")
    maxPooling2dLayer(2, Stride=2, Name="b1_pool")

    blocSeparable(c(3), "b2")
    maxPooling2dLayer(2, Stride=2, Name="b2_pool")
    blocSeparable(c(4), "b3")
    maxPooling2dLayer(2, Stride=2, Name="b3_pool")
    blocSeparable(c(5), "b4")
    maxPooling2dLayer(2, Stride=2, Name="b4_pool")
    blocSeparable(c(6), "b5")

    globalAveragePooling2dLayer(Name="gap")
    dropoutLayer(0.3, Name="dropout")
    fullyConnectedLayer(nbClasses, Name="fc")
    softmaxLayer(Name="softmax")
];
net = dlnetwork(couches);
end

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

function tracerCourbes(info, fichier)
% Perte et exactitude au fil des itérations, en entraînement et en validation.
try
    T = info.TrainingHistory;
    V = info.ValidationHistory;  V = V(V.Iteration > 0, :);
    fig = nouvelleFigure(16, 11);
    tiledlayout(fig, 2, 1, TileSpacing="compact", Padding="compact");
    motifs = ["loss" "accuracy"];  titres = ["Perte" "Exactitude (%)"];
    for k = 1:2
        ax = nexttile;  hold(ax, "on");
        cT = find(contains(lower(string(T.Properties.VariableNames)), motifs(k)), 1);
        cV = find(contains(lower(string(V.Properties.VariableNames)), motifs(k)), 1);
        if ~isempty(cT), plot(ax, T.Iteration, smoothdata(T{:, cT}, "movmean", 25), Color=couleur(1), LineWidth=1.2); end
        if ~isempty(cV), plot(ax, V.Iteration, V{:, cV}, "-o", Color=couleur(2), LineWidth=1.2, MarkerSize=4); end
        ylabel(ax, titres(k));
        mettreEnForme(ax);
    end
    xlabel(ax, "Itération");
    legend(ax, ["Entraînement" "Validation"], Location="best");
    exportgraphics(fig, fichier, Resolution=300);
catch err
    warning("Courbes non enregistrées : %s", err.message);
end
end

function tracerConfusion(yVrai, yPredit, fichier)
% Matrice de confusion avec le rappel par classe.
try
    fig = nouvelleFigure(16, 12);
    cm = confusionchart(fig, yVrai, yPredit, RowSummary="row-normalized");
    cm.Title = "";  cm.XLabel = "Classe prédite";  cm.YLabel = "Classe réelle";
    cm.FontName = "Times New Roman";  cm.FontSize = 10;
    exportgraphics(fig, fichier, Resolution=300);
catch err
    warning("Matrice de confusion non enregistrée : %s", err.message);
end
end

function fig = nouvelleFigure(largeurCm, hauteurCm)
fig = figure(Color="w", Units="centimeters", Position=[2 2 largeurCm hauteurCm]);
end

function mettreEnForme(ax)
set(ax, FontName="Times New Roman", FontSize=11, Box="off", TickLabelInterpreter="none");
grid(ax, "on");
end

function c = couleur(k)
palette = [0 0.447 0.741; 0.851 0.325 0.098];
c = palette(k, :);
end

function ecrirePredictions(baseDir, nom, F1, frug, fichiers, labels)
% Écrit les prédictions dans predictions/, sous un nom unique (date, modèle, scores).
dossier = fullfile(baseDir, "predictions");
if ~isfolder(dossier), mkdir(dossier); end
horodatage = string(datetime("now", Format="yyyyMMdd_HHmmss"));
f = fullfile(dossier, sprintf("pred_%s_%s_F1-%.4f_frug-%.4f.json", horodatage, nom, F1, frug));
fid = fopen(f, "w");
fprintf(fid, "{\n");
n = numel(fichiers);
for i = 1:n
    [~, image] = fileparts(fichiers{i});
    sep = ",";  if i == n, sep = ""; end
    fprintf(fid, '    "%s": "%s"%s\n', image, lower(string(labels(i))), sep);
end
fprintf(fid, "}\n");
fclose(fid);
fprintf("Prédictions écrites : %s\n", f);
end
