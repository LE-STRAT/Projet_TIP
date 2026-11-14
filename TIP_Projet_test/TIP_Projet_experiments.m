%% Food-11 : banc d'essai (enchaîne les configurations et compare F1, frugalité, durée)
% Chaque essai est entraîné, évalué sur la validation, puis ajouté à resultats.csv.
% Un essai qui échoue (package manquant, par exemple) est noté et le script continue.
clear; clc; close all;

%% 0. Réglages généraux
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
trainDir = fullfile(baseDir, "food-11", "train");
testDir  = fullfile(baseDir, "food-11", "test");
outDir   = fullfile(baseDir, "resultats_" + string(datetime("now", Format="yyyyMMdd_HHmm")));
mkdir(outDir);

hasGPU = canUseGPU;
tiersToRun = [1 2];                  % paliers lancés sans GPU
if hasGPU, tiersToRun = 4; end       % avec GPU : optimisation par étapes
maxMinutes    = 240;                 % budget total en minutes

% Essais déjà faits, sautés à la relance (strings(0) pour tout relancer)
essaisDejaFaits = ["sqz_96" "sqz_128" "res18_96" "res18_128_base" "res18_128_letterbox" ...
                   "res18_128_aug" "res18_128_lr3e-4" "res18_224_lb_aug"];
meilleureFrugaliteConnue = 0.6311;   % meilleure frugalité obtenue jusque-là
trainFraction = 1.0;                 % < 1 pour un test rapide sur une partie des images
writePredictions = true;             % un JSON par essai dans predictions/

fprintf("GPU détecté : %d | paliers : %s | sortie : %s\n", hasGPU, mat2str(tiersToRun), outDir);

%% 1. Liste des essais
% addExp(E, palier, nom, réseau, taille, epochs, batch, lr, augmentation, letterbox, palier de LR, options)
%   options : seed, bestVal (garde la meilleure epoch), copyBest (reprend la meilleure recette de B/C)
E = [];
% Palier 1 : réseaux légers, petites images
E = addExp(E,1,"sqz_96",       "squeezenet",  96,  3, 64, 1e-3, false, false, false);
E = addExp(E,1,"sqz_128",      "squeezenet",  128, 3, 64, 1e-3, false, false, false);
E = addExp(E,1,"mobv2_128",    "mobilenetv2", 128, 3, 64, 1e-3, false, false, false);
E = addExp(E,1,"res18_96",     "resnet18",    96,  3, 64, 1e-3, false, false, false);
% Palier 2 : un seul réglage change par rapport à res18_128_base
E = addExp(E,2,"res18_128_base",      "resnet18", 128, 3, 64, 1e-3, false, false, false);
E = addExp(E,2,"res18_128_letterbox", "resnet18", 128, 3, 64, 1e-3, false, true,  false);
E = addExp(E,2,"res18_128_aug",       "resnet18", 128, 3, 64, 1e-3, true,  false, false);
E = addExp(E,2,"res18_128_lr3e-4",    "resnet18", 128, 3, 64, 3e-4, false, false, true);
E = addExp(E,2,"sqz_128_lr3e-4",      "squeezenet", 128, 3, 64, 3e-4, false, false, true);  % squeezenet était instable avec lr 1e-3
% Palier 3 : première configuration complète en 224 px
E = addExp(E,3,"res18_224_lb_aug",    "resnet18", 224, 10, 64, 3e-4, true, true, true);

% Palier 4 : optimisation par étapes. Les NaN sont remplis à partir des étapes précédentes.
% A : learning rate (224 px, 6 epochs, sans augmentation ni letterbox)
E = addExp(E,4,"A_lr1e-4", "resnet18", 224, 6, 64, 1e-4, false, false, true);
E = addExp(E,4,"A_lr3e-4", "resnet18", 224, 6, 64, 3e-4, false, false, true);
E = addExp(E,4,"A_lr1e-3", "resnet18", 224, 6, 64, 1e-3, false, false, true);
% B : entraînement long avec le meilleur LR de A, pour repérer la meilleure epoch
E = addExp(E,4,"B_long_20ep", "resnet18", 224, 20, 64, NaN, false, false, true, bestVal=true);
% C : augmentation et letterbox, avec le LR de A et (meilleure epoch de B + 5) epochs
E = addExp(E,4,"C_aug",    "resnet18", 224, NaN, 64, NaN, true,  false, true, bestVal=true);
E = addExp(E,4,"C_lb",     "resnet18", 224, NaN, 64, NaN, false, true,  true, bestVal=true);
E = addExp(E,4,"C_lb_aug", "resnet18", 224, NaN, 64, NaN, true,  true,  true, bestVal=true);
% D : meilleure recette de B/C avec une autre graine
E = addExp(E,4,"D_meilleur_graine2", "resnet18", 224, NaN, 64, NaN, false, false, true, ...
    bestVal=true, copyBest=true, seed=2);
% E : autres réseaux avec cette même recette
E = addExp(E,4,"E_effb0",  "efficientnetb0", 224, NaN, 64, NaN, false, false, true, bestVal=true, copyBest=true);
E = addExp(E,4,"E_mobv2",  "mobilenetv2",    224, NaN, 64, NaN, false, false, true, bestVal=true, copyBest=true);
E = addExp(E,4,"E_sqz",    "squeezenet",     224, NaN, 64, NaN, false, false, true, bestVal=true, copyBest=true);
E = addExp(E,4,"E_res50",  "resnet50",       224, NaN, 64, NaN, false, false, true, bestVal=true, copyBest=true);

%% 2. Données (même découpage 80/20 pour tous les essais)
imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
rng(0);
if trainFraction < 1
    imds = splitEachLabel(imds, trainFraction, "randomized");
end
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
imdsTest = imageDatastore(testDir);
fprintf("%d images d'entraînement, %d de validation, %d classes\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

%% 3. Boucle sur tous les essais
results = table();
bestFrug = -inf; bestNet = []; bestCfg = [];
epochsOpt = 10;                      % remplacé par la meilleure epoch trouvée à l'étape B
tStart = tic;
nPlanned = sum(ismember([E.tier], tiersToRun) & ~ismember([E.name], essaisDejaFaits));
iRun = 0;
figSum = figure(Name="Banc d'essai TIP : récapitulatif", NumberTitle="off", Position=[80 80 1100 650]);
annotation(figSum, "textbox", [0 0.45 1 0.1], String="En attente du premier essai...", ...
    HorizontalAlignment="center", EdgeColor="none", FontSize=14);

for k = 1:numel(E)
    c = E(k);
    if ~ismember(c.tier, tiersToRun), continue; end
    if ismember(c.name, essaisDejaFaits)
        fprintf("(essai %s déjà fait : sauté)\n", c.name);
        continue
    end
    % Réglages automatiques (étapes B à E)
    if c.copyBest
        ref = pickBest(results);
        if ~isempty(ref)
            c.lr = ref.lr; c.epochs = ref.epochs; c.augment = ref.augmentation;
            c.letterbox = ref.letterbox; c.sched = ref.lrSchedule;
            fprintf("Recette copiée de %s\n", ref.nom);
        end
    end
    if isnan(c.lr), c.lr = pickLR(results); end
    if isnan(c.epochs), c.epochs = min(20, epochsOpt + 5); end
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
        "exactitude",NaN, "frugalite",NaN, "graine",c.seed, "meilleureEpoch",NaN, ...
        "minutes",NaN, "statut","");
    try
        tRun = tic;
        rng(c.seed);                     % graine propre à chaque essai
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
        if c.bestVal, opts.OutputNetwork = "best-validation"; end
        [net, info] = trainnet(trainDS, net, "crossentropy", opts);
        try
            vh = info.ValidationHistory;
            vh = vh(vh.Iteration > 0, :);
            [~, ib] = min(vh.Loss);
            row.meilleureEpoch = max(1, round(vh.Iteration(ib) / valFreq));
        catch
        end
        if startsWith(c.name, "B_") && ~isnan(row.meilleureEpoch)
            epochsOpt = max(3, row.meilleureEpoch);
            fprintf("-> Étape B : meilleure epoch = %d (étape C entraînée sur %d epochs)\n", ...
                epochsOpt, min(20, epochsOpt + 5));
        end

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

        if writePredictions
            try
                YTest = scores2label(minibatchpredict(net, makeDS(imdsTest, c, false, false)), classNames);
                savePredictions(baseDir, "bancessai_" + c.name, F1, frug, imdsTest.Files, YTest);
            catch MEp
                fprintf("(prédictions du test non écrites : %s)\n", MEp.message);
            end
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
    figSum = updateSummary(figSum, results, iRun, nPlanned, toc(tStart)/60, maxMinutes);
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
if any(startsWith(results.nom, "A_"))
    fprintf("LR retenu (étape A) : %g | meilleure epoch (étape B) : %d\n", pickLR(results), epochsOpt);
end
fprintf("Temps total : %.1f min | fichiers dans %s\n", toc(tStart)/60, outDir);

%% 5. Comparaison avec la référence
if ~isempty(bestNet) && bestFrug <= meilleureFrugaliteConnue
    fprintf("Remarque : aucun essai ne bat la frugalité de référence (%.4f).\n", meilleureFrugaliteConnue);
elseif ~isempty(bestNet)
    fprintf("Nouvelle meilleure frugalité : %s (%.4f > %.4f)\n", bestCfg.name, bestFrug, meilleureFrugaliteConnue);
end

%% Fonctions locales
function E = addExp(E, tier, name, netName, sz, epochs, batch, lr, aug, letterbox, sched, opt)
arguments
    E; tier; name; netName; sz; epochs; batch; lr; aug; letterbox; sched
    opt.seed = 0             % graine aléatoire de l'essai
    opt.bestVal = false      % garde le réseau de la meilleure epoch (validation)
    opt.copyBest = false     % reprend la meilleure recette des étapes B/C
end
s = struct("tier",tier, "name",name, "netName",netName, "inputSize",sz, "epochs",epochs, ...
    "batch",batch, "lr",lr, "augment",aug, "letterbox",letterbox, "sched",sched, ...
    "seed",opt.seed, "bestVal",opt.bestVal, "copyBest",opt.copyBest);
if isempty(E), E = s; else, E(end+1) = s; end %#ok<AGROW>
end

function lr = pickLR(results)
% Meilleur learning rate de l'étape A (selon le F1). 3e-4 par défaut.
lr = 3e-4;
if isempty(results), return; end
A = results(startsWith(results.nom, "A_") & results.statut == "ok", :);
if ~isempty(A)
    [~, i] = max(A.F1);
    lr = A.lr(i);
end
end

function ref = pickBest(results)
% Meilleure recette (F1) parmi les étapes B et C. Vide si aucune n'a réussi.
ref = [];
if isempty(results), return; end
R = results(startsWith(results.nom, ["B_" "C_"]) & results.statut == "ok", :);
if ~isempty(R)
    [~, i] = max(R.F1);
    ref = R(i, :);
end
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

function fig = updateSummary(fig, results, iRun, nPlanned, elapsed, maxMinutes)
% Fenêtre de synthèse mise à jour après chaque essai (rouverte si elle a été fermée).
if ~isvalid(fig)
    fig = figure(Name="Banc d'essai TIP : récapitulatif", NumberTitle="off", Position=[80 80 1100 650]);
end
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
