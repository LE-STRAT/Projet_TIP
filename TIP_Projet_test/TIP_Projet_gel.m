%% TIP Food-11 : gel des couches (frozen) vs fine-tuning, avec ResNet-18
% Compare plusieurs stratégies, du réseau entièrement gelé au réseau entièrement réentraîné.
% Tout le reste est identique au modèle final (224 px, letterbox, augmentation,
% Adam 3e-4 divisé par 10 à mi-parcours, 10 epochs, batch 64), pour que seul le gel change.
%
% Structure de ResNet-18 (du début à la fin) :
%   conv1 -> bloc res2 -> bloc res3 -> bloc res4 -> bloc res5 (= res5a + res5b) -> couche finale (fc, 11 classes)
%   Le bloc res5 contient à lui seul environ 75 % des paramètres du réseau.
%
% Prérequis : Deep Learning Toolbox, package ResNet-18, Image Processing Toolbox. GPU très conseillé
% (environ 1 h pour les 6 essais sur GPU ; sur CPU, plusieurs heures PAR essai).
clear; clc; close all;

%% 0. Réglages
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");
outDir   = fullfile(baseDir, "resultats_gel_" + string(datetime("now", Format="yyyyMMdd_HHmm")));
mkdir(outDir);

netName = "resnet18"; inputSize = 224; maxEpochs = 10; batchSize = 64;
maxMinutes = 240;                    % plus aucun nouvel essai n'est lancé après ce temps total

% Stratégies testées. trainFrom = à partir d'où on réentraîne (tout ce qui est avant est gelé)
%   "tete"  : seule la nouvelle couche finale est entraînée (réseau entièrement gelé)
%   "res5b" : dernier sous-bloc + couche finale
%   "res5"  : dernier bloc + couche finale
%   "res4"  : deux derniers blocs + couche finale
%   "tout"  : fine-tuning complet (= le modèle final actuel, pour comparer dans les mêmes conditions)
V = struct("nom",{}, "trainFrom",{}, "lr",{});
V(end+1) = struct("nom","1_gel_total",         "trainFrom","tete",  "lr",3e-4);
V(end+1) = struct("nom","2_gel_total_lr1e-3",  "trainFrom","tete",  "lr",1e-3); % une couche neuve seule supporte un LR plus fort
V(end+1) = struct("nom","3_dernier_sous_bloc", "trainFrom","res5b", "lr",3e-4);
V(end+1) = struct("nom","4_dernier_bloc",      "trainFrom","res5",  "lr",3e-4);
V(end+1) = struct("nom","5_deux_derniers_blocs","trainFrom","res4", "lr",3e-4);
V(end+1) = struct("nom","6_tout_reentraine",   "trainFrom","tout",  "lr",3e-4);

if canUseGPU
    fprintf("GPU détecté. Sortie : %s\n", outDir);
else
    warning("Aucun GPU détecté : chaque essai prendra plusieurs heures sur CPU.");
end

%% 1. Données (même découpage que le banc d'essai)
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
trainDS = combine(transform(imdsTrain, @(x) {padResize(x, inputSize, true)}),  arrayDatastore(imdsTrain.Labels));
valDS   = combine(transform(imdsVal,   @(x) {padResize(x, inputSize, false)}), arrayDatastore(imdsVal.Labels));
valImgs = transform(imdsVal, @(x) {padResize(x, inputSize, false)});
imdsTest = imageDatastore(testDir);
testImgs = transform(imdsTest, @(x) {padResize(x, inputSize, false)});
itersPerEpoch = max(1, floor(numel(imdsTrain.Files)/batchSize));

%% 2. Boucle sur les stratégies
results = table();
bestF1 = -inf; bestNet = []; bestName = "";
figSum = figure(Name="Gel des couches : récapitulatif", NumberTitle="off", Position=[80 80 1000 600]);
tStart = tic;

for k = 1:numel(V)
    v = V(k);
    if toc(tStart)/60 > maxMinutes
        fprintf("Budget de %d min atteint : arrêt avant %s.\n", maxMinutes, v.nom); break
    end
    fprintf("\n=== [%d/%d] %s (réentraîné à partir de : %s, LR %g) ===\n", k, numel(V), v.nom, v.trainFrom, v.lr);
    row = struct("nom",string(v.nom), "reentraineDepuis",string(v.trainFrom), "lr",v.lr, ...
        "paramsTotalM",NaN, "paramsEntrainesM",NaN, "partEntrainee",NaN, "F1",NaN, ...
        "exactitude",NaN, "frugalite",NaN, "minutes",NaN, "statut","");
    tRun = tic;
    try
        rng(1);                          % même initialisation de la couche finale pour toutes les stratégies
        net = imagePretrainedNetwork(netName, NumClasses=numClasses);
        nTotal = sum(cellfun(@numel, net.Learnables.Value));
        [net, nTrain] = freezeBefore(net, v.trainFrom);
        fprintf("Paramètres entraînés : %.3f M sur %.2f M (%.1f %%)\n", nTrain/1e6, nTotal/1e6, 100*nTrain/nTotal);

        opts = trainingOptions("adam", ...
            InitialLearnRate=v.lr, MiniBatchSize=batchSize, MaxEpochs=maxEpochs, ...
            LearnRateSchedule="piecewise", LearnRateDropFactor=0.1, LearnRateDropPeriod=floor(maxEpochs/2), ...
            Shuffle="every-epoch", ValidationData=valDS, ValidationFrequency=itersPerEpoch, ...
            Metrics="accuracy", Plots="training-progress", Verbose=true, VerboseFrequency=100, ...
            ExecutionEnvironment="auto");
        net = trainnet(trainDS, net, "crossentropy", opts);

        YPred = scores2label(minibatchpredict(net, valImgs), classNames);
        YTrue = imdsVal.Labels;
        F1 = macroF1(YTrue, YPred);
        R = nTotal/1e6;                  % la frugalité compte TOUS les paramètres, gelés ou non
        row.paramsTotalM = R; row.paramsEntrainesM = nTrain/1e6; row.partEntrainee = nTrain/nTotal;
        row.F1 = F1; row.exactitude = mean(YPred == YTrue); row.frugalite = F1 - 0.3/(1 + 1/R);
        row.statut = "ok";
        fprintf("-> F1 = %.4f | Frugalité = %.4f | %.1f min\n", F1, row.frugalite, toc(tRun)/60);

        try
            fig = figure(Visible="off"); confusionchart(YTrue, YPred);
            exportgraphics(fig, fullfile(outDir, v.nom + "_confusion.png")); close(fig);
        catch
        end
        YTest = scores2label(minibatchpredict(net, testImgs), classNames);
        savePredictions(baseDir, "gel_" + string(v.nom), F1, row.frugalite, imdsTest.Files, YTest);
        if F1 > bestF1, bestF1 = F1; bestNet = net; bestName = string(v.nom); end
    catch ME
        row.statut = "ERREUR : " + string(strtok(ME.message, newline));
        fprintf("-> échec : %s\n", ME.message);
    end
    row.minutes = toc(tRun)/60;
    results = [results; struct2table(row)]; %#ok<AGROW>
    writetable(results, fullfile(outDir, "resultats_gel.csv"));
    figSum = drawSummary(figSum, results);
end

%% 3. Récapitulatif
fprintf("\n================ RÉCAPITULATIF ================\n");
disp(results(:, ["nom" "partEntrainee" "F1" "exactitude" "frugalite" "minutes" "statut"]));
if isvalid(figSum), exportgraphics(figSum, fullfile(outDir, "recapitulatif_gel.png")); end

if ~isempty(bestNet)
    fprintf("Meilleure stratégie : %s (F1 = %.4f). Un JSON par stratégie est dans le dossier predictions/.\n", bestName, bestF1);
end

%% ===================== Fonctions locales =====================
function [net, nTrain] = freezeBefore(net, trainFrom)
% Gèle (learning rate x 0) toutes les couches situées avant le point "trainFrom".
seuils = struct("tete",99, "res5b",51, "res5",50, "res4",40, "tout",0);
T = seuils.(char(trainFrom));
L = net.Learnables;
nTrain = 0;
for i = 1:height(L)
    if layerUnit(L.Layer(i)) < T
        net = setLearnRateFactor(net, L.Layer(i), L.Parameter(i), 0);
    else
        nTrain = nTrain + numel(L.Value{i});
    end
end
if T == 99 && nTrain > 1e5
    warning("Noms de couches inattendus : le gel ne s'est peut-être pas appliqué correctement.");
end
end

function u = layerUnit(name)
% Position d'une couche dans ResNet-18 : conv1 = 10, res2a = 20, res2b = 21, ..., res5b = 51, couche finale = 99
name = char(name);
tok = regexp(name, '^(?:res|bn)(\d)([a-z])', 'tokens', 'once');
if ~isempty(tok)
    u = 10*str2double(tok{1}) + (tok{2}(1) - 'a');
elseif contains(name, "conv1")
    u = 10;
else
    u = 99;
end
end

function F1 = macroF1(YTrue, YPred)
C = confusionmat(YTrue, YPred);
prec = diag(C) ./ sum(C,1)';
rec  = diag(C) ./ sum(C,2);
F1 = mean(2*prec.*rec ./ (prec+rec), "omitnan");
end

function fig = drawSummary(fig, results)
if ~isvalid(fig)
    fig = figure(Name="Gel des couches : récapitulatif", NumberTitle="off", Position=[80 80 1000 600]);
end
clf(fig);
names = categorical(results.nom); names = reordercats(names, cellstr(results.nom));
tl = tiledlayout(fig, 2, 1, TileSpacing="compact");
title(tl, "Gel des couches : de « tout gelé » à « tout réentraîné »", FontWeight="bold");
ax1 = nexttile(tl);
bar(ax1, names, [results.F1 results.frugalite]);
legend(ax1, ["F1 macro" "Frugalité"], Location="northoutside", Orientation="horizontal");
grid(ax1, "on"); ylabel(ax1, "Score"); set(ax1, TickLabelInterpreter="none");
ax2 = nexttile(tl);
bar(ax2, names, 100*results.partEntrainee);
ylabel(ax2, "% des paramètres réentraînés"); grid(ax2, "on"); set(ax2, TickLabelInterpreter="none");
drawnow;
end

function out = padResize(img, sz, doAug)
% Letterbox (+ miroir et rotation aléatoires si doAug), identique au modèle final.
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
