%% TIP Food-11 : banc d'essai automatique (lance tous les entraînements d'un coup)
% Utilisation : ouvrir ce fichier, régler la section 0 si besoin, puis Run (F5).
% Chaque essai est entraîné, évalué (F1 macro, frugalité), puis ajouté à
% resultats.csv. Si un essai échoue (ex. support package manquant), il est
% noté dans le tableau et le script passe au suivant.
clear; clc; close all;

%% 0. Réglages généraux
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");
outDir   = fullfile(baseDir, "resultats_" + string(datetime("now", Format="yyyyMMdd_HHmm")));
mkdir(outDir);

hasGPU = canUseGPU;                  % true si GPU NVIDIA + Parallel Computing Toolbox
tiersToRun = [1 2];                  % 1 = simple, 2 = intermédiaire, 3 = élaboré (GPU conseillé)
if hasGPU, tiersToRun = [1 2 3]; end
maxMinutes    = 120;                 % plus aucun nouvel essai n'est lancé après ce temps total
trainFraction = 1.0;                 % ex. 0.3 pour tester le script vite sur 30 % des images
writePredictions = true;             % écrit predictions_best.json avec le meilleur modèle (frugalité)

fprintf("GPU détecté : %d | paliers : %s | sortie : %s\n", hasGPU, mat2str(tiersToRun), outDir);

%% 1. Liste des essais : du plus simple au plus élaboré
% addExp(E, palier, nom, réseau, taille, epochs, batch, lr, augmentation, letterbox, lrSchedule)
%   letterbox = true : image entière conservée (bandes noires), sans déformation
%   letterbox = false: image étirée en carré (comme le script d'origine)
E = [];
% --- Palier 1 : rapide, réseaux légers ---
E = addExp(E,1,"sqz_96",       "squeezenet",  96,  3, 64, 1e-3, false, false, false);
E = addExp(E,1,"sqz_128",      "squeezenet",  128, 3, 64, 1e-3, false, false, false);
E = addExp(E,1,"mobv2_128",    "mobilenetv2", 128, 3, 64, 1e-3, false, false, false);
E = addExp(E,1,"res18_96",     "resnet18",    96,  3, 64, 1e-3, false, false, false);
% --- Palier 2 : on teste letterbox, augmentation, learning rate ---
E = addExp(E,2,"res18_128_base",      "resnet18", 128, 3, 64, 1e-3, false, false, false);
E = addExp(E,2,"res18_128_letterbox", "resnet18", 128, 3, 64, 1e-3, false, true,  false);
E = addExp(E,2,"res18_128_aug",       "resnet18", 128, 3, 64, 1e-3, true,  false, false);
E = addExp(E,2,"res18_128_lr3e-4",    "resnet18", 128, 3, 64, 3e-4, false, false, true);
% --- Palier 3 : élaboré, pour GPU (224 px, plus d'epochs, réseaux plus gros) ---
E = addExp(E,3,"res18_224_lb_aug",    "resnet18",       224, 10, 64, 3e-4, true, true, true);
E = addExp(E,3,"effb0_224_lb_aug",    "efficientnetb0", 224, 10, 64, 3e-4, true, true, true);
E = addExp(E,3,"res50_224_lb_aug",    "resnet50",       224, 10, 64, 3e-4, true, true, true);
E = addExp(E,3,"mobv2_224_lb_aug",    "mobilenetv2",    224, 15, 64, 3e-4, true, true, true);

%% 2. Données (même découpage 80/20 pour tous les essais)
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
if trainFraction < 1
    imds = splitEachLabel(imds, trainFraction, "randomized");
end
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("%d images d'entraînement, %d de validation, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

%% 3. Boucle sur tous les essais
results = table();
bestFrug = -inf; bestNet = []; bestCfg = [];
tStart = tic;
nPlanned = sum(ismember([E.tier], tiersToRun));
iRun = 0;
figSum = figure(Name="Banc d'essai TIP : récapitulatif", NumberTitle="off", Position=[80 80 1100 650]);
annotation(figSum, "textbox", [0 0.45 1 0.1], String="En attente du premier essai...", ...
    HorizontalAlignment="center", EdgeColor="none", FontSize=14);

for k = 1:numel(E)
    c = E(k);
    if ~ismember(c.tier, tiersToRun), continue; end
    if toc(tStart)/60 > maxMinutes
        fprintf("\nBudget de %d min atteint : arrêt avant l'essai %s.\n", maxMinutes, c.name);
        break
    end
    iRun = iRun + 1;
    if isvalid(figSum)
        figSum.Name = sprintf("Banc d'essai TIP : essai %d/%d en cours (%s)", iRun, nPlanned, c.name);
        drawnow;
    end
    fprintf("\n=== [%d/%d] %s (%s, %dpx, %d epochs, lr=%g, aug=%d, letterbox=%d) ===\n", ...
        k, numel(E), c.name, c.netName, c.inputSize, c.epochs, c.lr, c.augment, c.letterbox);

    row = struct("nom",string(c.name), "reseau",string(c.netName), "taille",c.inputSize, ...
        "epochs",c.epochs, "batch",c.batch, "lr",c.lr, "augmentation",c.augment, ...
        "letterbox",c.letterbox, "lrSchedule",c.sched, "paramsM",NaN, "F1",NaN, ...
        "exactitude",NaN, "frugalite",NaN, "minutes",NaN, "statut","");
    try
        tRun = tic;
        [net, nParams] = buildNet(c.netName, numClasses);
        trainDS = makeDS(imdsTrain, c, true,  true);
        valDS   = makeDS(imdsVal,   c, false, true);
        valFreq = max(1, floor(numel(imdsTrain.Files)/c.batch));

        sched = "none"; if c.sched, sched = "piecewise"; end
        opts = trainingOptions("adam", ...
            InitialLearnRate=c.lr, MiniBatchSize=c.batch, MaxEpochs=c.epochs, ...
            Shuffle="every-epoch", ValidationData=valDS, ValidationFrequency=valFreq, ...
            Metrics="accuracy", Plots="training-progress", Verbose=true, VerboseFrequency=100, ...
            LearnRateSchedule=sched, LearnRateDropFactor=0.1, ...
            LearnRateDropPeriod=max(1, floor(c.epochs/2)), ExecutionEnvironment="auto");
        net = trainnet(trainDS, net, "crossentropy", opts);

        % Évaluation sur la validation
        scores = minibatchpredict(net, makeDS(imdsVal, c, false, false));
        YPred = scores2label(scores, classNames);
        YTrue = imdsVal.Labels;
        C = confusionmat(YTrue, YPred);
        prec = diag(C) ./ sum(C,1)';
        rec  = diag(C) ./ sum(C,2);
        f1   = 2*prec.*rec ./ (prec+rec);
        F1   = mean(f1, "omitnan");
        R    = nParams/1e6;
        frug = F1 - 0.3/(1 + 1/R);

        row.paramsM = R; row.F1 = F1; row.exactitude = mean(YPred == YTrue);
        row.frugalite = frug; row.minutes = toc(tRun)/60; row.statut = "ok";
        fprintf("-> F1 = %.4f | Frugalité = %.4f | Params = %.2f M | %.1f min\n", ...
            F1, frug, R, row.minutes);

        try
            fig = figure(Visible="off");
            confusionchart(YTrue, YPred);
            exportgraphics(fig, fullfile(outDir, c.name + "_confusion.png"));
            close(fig);
        catch
        end

        if frug > bestFrug
            bestFrug = frug; bestNet = net; bestCfg = c;
            save(fullfile(outDir, "meilleur_modele.mat"), "bestNet", "bestCfg", "classNames", "-v7.3");
        end
    catch ME
        row.statut = "ERREUR : " + string(strtok(ME.message, newline));
        row.minutes = toc(tRun)/60;
        fprintf("-> échec : %s\n", ME.message);
    end
    results = [results; struct2table(row)]; %#ok<AGROW>
    writetable(results, fullfile(outDir, "resultats.csv"));   % sauvegarde après chaque essai
    save(fullfile(outDir, "resultats.mat"), "results");
    updateSummary(figSum, results, iRun, nPlanned, toc(tStart)/60, maxMinutes);
end
if isvalid(figSum)
    figSum.Name = "Banc d'essai TIP : terminé";
    exportgraphics(figSum, fullfile(outDir, "recapitulatif.png"));
end

%% 4. Récapitulatif
fprintf("\n================ RÉCAPITULATIF (trié par frugalité) ================\n");
ok = results(results.statut == "ok", :);
if ~isempty(ok)
    disp(sortrows(ok, "frugalite", "descend"));
    fprintf("Meilleur F1 : %s (%.4f) | Meilleure frugalité : %s (%.4f)\n", ...
        ok.nom(find(ok.F1 == max(ok.F1), 1)), max(ok.F1), bestCfg.name, bestFrug);
end
bad = results(results.statut ~= "ok", ["nom" "statut"]);
if ~isempty(bad), disp("Essais en échec :"); disp(bad); end
fprintf("Temps total : %.1f min | fichiers dans %s\n", toc(tStart)/60, outDir);

%% 5. Prédictions du test avec le meilleur modèle (fichier séparé de predictions.json)
if writePredictions && ~isempty(bestNet)
    imdsTest = imageDatastore(testDir);
    YTest = scores2label(minibatchpredict(bestNet, makeDS(imdsTest, bestCfg, false, false)), classNames);
    fid = fopen(fullfile(outDir, "predictions_best.json"), "w");
    fprintf(fid, "{\n");
    n = numel(imdsTest.Files);
    for i = 1:n
        [~, name] = fileparts(imdsTest.Files{i});
        sep = ","; if i == n, sep = ""; end
        fprintf(fid, '    "%s": "%s"%s\n', name, lower(string(YTest(i))), sep);
    end
    fprintf(fid, "}\n");
    fclose(fid);
    fprintf("predictions_best.json écrit (modèle : %s)\n", bestCfg.name);
end

%% ===================== Fonctions locales =====================
function E = addExp(E, tier, name, netName, sz, epochs, batch, lr, aug, letterbox, sched)
s = struct("tier",tier, "name",name, "netName",netName, "inputSize",sz, "epochs",epochs, ...
    "batch",batch, "lr",lr, "augment",aug, "letterbox",letterbox, "sched",sched);
if isempty(E), E = s; else, E(end+1) = s; end %#ok<AGROW>
end

function [net, nParams] = buildNet(netName, numClasses)
net = imagePretrainedNetwork(netName, NumClasses=numClasses);
nParams = sum(cellfun(@numel, net.Learnables.Value));
end

function ds = makeDS(imds, c, isTrain, withLabels)
sz = c.inputSize;
if c.letterbox
    doAug = isTrain && c.augment;
    imgDS = transform(imds, @(x) {padResize(x, sz, doAug)});
    if withLabels
        ds = combine(imgDS, arrayDatastore(imds.Labels));
    else
        ds = imgDS;
    end
else
    if isTrain && c.augment
        aug = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15]);
        ds = augmentedImageDatastore([sz sz], imds, ColorPreprocessing="gray2rgb", DataAugmentation=aug);
    else
        ds = augmentedImageDatastore([sz sz], imds, ColorPreprocessing="gray2rgb");
    end
end
end

function out = padResize(img, sz, doAug)
% Redimensionne en gardant le rapport hauteur/largeur, puis centre sur un carré noir.
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

function updateSummary(fig, results, iRun, nPlanned, elapsed, maxMinutes)
% Fenêtre de synthèse mise à jour après chaque essai.
if ~isvalid(fig), return; end
clf(fig);
names = categorical(results.nom);
names = reordercats(names, cellstr(results.nom));
tl = tiledlayout(fig, 2, 2, TileSpacing="compact", Padding="compact");
title(tl, sprintf("Essai %d / %d terminé  |  %.0f min écoulées (budget %g min)", ...
    iRun, nPlanned, elapsed, maxMinutes), FontWeight="bold");

ax1 = nexttile(tl, [1 2]);
bar(ax1, names, [results.F1 results.frugalite]);
legend(ax1, ["F1 macro" "Frugalité"], Location="northoutside", Orientation="horizontal");
ylabel(ax1, "Score"); grid(ax1, "on");
title(ax1, "Scores par essai (barre absente = essai en erreur)");

ok = results.statut == "ok";
ax2 = nexttile(tl);
if any(ok)
    scatter(ax2, results.paramsM(ok), results.F1(ok), 60, "filled");
    text(ax2, results.paramsM(ok), results.F1(ok), "  " + results.nom(ok), FontSize=8, Interpreter="none");
end
xlabel(ax2, "Paramètres (M)"); ylabel(ax2, "F1 macro"); grid(ax2, "on");
title(ax2, "Précision selon la taille du réseau");

ax3 = nexttile(tl);
barh(ax3, names, results.minutes);
xlabel(ax3, "Durée (min)"); grid(ax3, "on");
title(ax3, "Temps d'entraînement");
set([ax1 ax3], TickLabelInterpreter="none");
drawnow;
end
