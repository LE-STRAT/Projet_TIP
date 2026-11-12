%% ========================================================================
%  Food-11 : CNN "maison" frugal (Matlab R2024a+, Deep Learning Toolbox)
%  Score de frugalité : Frug = F1 - 0.3 * (R / (1 + R)),  R = nb params (M)
%% ========================================================================
clear; clc; close all;
rng(42);                                    % reproductibilité

%% ---------------------------- 1. PARAMÈTRES ------------------------------
dossierTrain = "food-11/train";                     % 11 sous-dossiers = 11 classes
dossierTest  = "food-11/test";                      % images sans labels
tailleImg    = [96 96 3];                   % essayer [128 128 3] si besoin
canaux       = [16 32 64 128 256 256];      % largeur des blocs (levier de frugalité)
nbEpoques    = 40;
tailleBatch  = 64;
lrInitial    = 2e-3;

%% ------------------------- 2. DONNÉES (imageDatastore) -------------------
imds = imageDatastore(dossierTrain, ...
    IncludeSubfolders=true, LabelSource="foldernames");

classes    = categories(imds.Labels);       % ordre alphabétique des dossiers
nbClasses  = numel(classes);                % doit valoir 11
fprintf("Classes détectées (%d) : %s\n", nbClasses, strjoin(string(classes), ", "));

% Séparation 80% train / 20% validation (stratifiée par classe)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
fprintf("Train : %d images | Validation : %d images\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files));

% Augmentation (uniquement sur le train) : ne coûte AUCUN paramètre
augmenteur = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-20 20], ...
    RandScale=[0.85 1.15], ...
    RandXTranslation=[-8 8], ...
    RandYTranslation=[-8 8]);

% augmentedImageDatastore redimensionne + convertit les niveaux de gris en RGB
augTrain = augmentedImageDatastore(tailleImg(1:2), imdsTrain, ...
    DataAugmentation=augmenteur, ColorPreprocessing="gray2rgb");
augVal   = augmentedImageDatastore(tailleImg(1:2), imdsVal, ...
    ColorPreprocessing="gray2rgb");

%% ------------------------- 3. ARCHITECTURE FRUGALE -----------------------
c = canaux;
couches = [
    imageInputLayer(tailleImg, Normalization="none", Name="entree")
    % La BatchNorm juste après la 1re conv gère l'échelle 0-255 des pixels

    % --- Stem : conv 3x3 classique (peu de filtres) ---
    convolution2dLayer(3, c(1), Padding="same", Name="stem_conv")
    batchNormalizationLayer(Name="stem_bn")
    reluLayer(Name="stem_relu")
    maxPooling2dLayer(2, Stride=2, Name="stem_pool")            % 96 -> 48

    % --- Bloc 1 : conv 3x3 classique ---
    convolution2dLayer(3, c(2), Padding="same", Name="b1_conv")
    batchNormalizationLayer(Name="b1_bn")
    reluLayer(Name="b1_relu")
    maxPooling2dLayer(2, Stride=2, Name="b1_pool")              % 48 -> 24

    % --- Blocs 2 à 5 : convolutions séparables (depthwise + pointwise) ---
    blocSeparable(c(3), "b2")
    maxPooling2dLayer(2, Stride=2, Name="b2_pool")              % 24 -> 12
    blocSeparable(c(4), "b3")
    maxPooling2dLayer(2, Stride=2, Name="b3_pool")              % 12 -> 6
    blocSeparable(c(5), "b4")
    maxPooling2dLayer(2, Stride=2, Name="b4_pool")              % 6 -> 3
    blocSeparable(c(6), "b5")

    % --- Tête de classification ultra-légère ---
    globalAveragePooling2dLayer(Name="gap")                     % pas de flatten !
    dropoutLayer(0.3, Name="dropout")
    fullyConnectedLayer(nbClasses, Name="fc")
    softmaxLayer(Name="softmax")
];

net = dlnetwork(couches);

% Comptage des paramètres apprenables (poids, biais, scale/offset des BN)
nbParams = sum(cellfun(@numel, net.Learnables.Value));
R = nbParams / 1e6;                                             % en millions
fprintf("\nNombre de paramètres : %d (R = %.4f M)\n", nbParams, R);
fprintf("Pénalité de frugalité : %.4f\n", 0.3 * (R / (1 + R)));
if nbParams < 5e4 || nbParams > 5e5
    warning("Nombre de paramètres hors de la plage visée (50k - 500k).");
end

%% ------------------------------ 4. ENTRAÎNEMENT --------------------------
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
    OutputNetwork="best-validation", ...     % garde le meilleur modèle
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ExecutionEnvironment="auto");            % GPU si disponible

net = trainnet(augTrain, net, "crossentropy", options);

save("cnn_frugal_food11.mat", "net", "classes", "tailleImg");

%% ------------------- 5. ÉVALUATION : F1 MACRO + FRUGALITÉ ----------------
% Datastore de validation SANS labels pour la prédiction (même ordre que imdsVal)
augValPred = augmentedImageDatastore(tailleImg(1:2), ...
    imageDatastore(imdsVal.Files), ColorPreprocessing="gray2rgb");

scoresVal = minibatchpredict(net, augValPred, MiniBatchSize=128);
predVal   = scores2label(scoresVal, classes);               % catégorique

% Matrice de confusion (sans toolbox Statistics) : lignes = vrai, colonnes = prédit
yVrai = double(imdsVal.Labels);
yPred = double(categorical(predVal, classes));
C = accumarray([yVrai yPred], 1, [nbClasses nbClasses]);

vp        = diag(C);
precision = vp ./ sum(C, 1)';                               % par classe
rappel    = vp ./ sum(C, 2);
f1Classe  = 2 * precision .* rappel ./ (precision + rappel);
f1Classe(isnan(f1Classe)) = 0;                              % classes sans prédiction
f1Macro   = mean(f1Classe);

frug = f1Macro - 0.3 * (R / (1 + R));

fprintf("\n================ RÉSULTATS (validation) ================\n");
fprintf("Accuracy        : %.4f\n", sum(vp) / sum(C, "all"));
fprintf("F1 macro        : %.4f\n", f1Macro);
fprintf("Paramètres      : %d (R = %.4f M)\n", nbParams, R);
fprintf("Score frugalité : %.4f\n", frug);
fprintf("=========================================================\n");
disp(table(classes, round(f1Classe, 3), VariableNames=["Classe" "F1"]));

figure; confusionchart(imdsVal.Labels, predVal, ...
    RowSummary="row-normalized", Title="Matrice de confusion (validation)");

%% ------------------------ 6. INFÉRENCE SUR LE DOSSIER TEST ---------------
imdsTest = imageDatastore(dossierTest);
augTest  = augmentedImageDatastore(tailleImg(1:2), imdsTest, ...
    ColorPreprocessing="gray2rgb");

scoresTest = minibatchpredict(net, augTest, MiniBatchSize=128);
predTest   = scores2label(scoresTest, classes);

% Noms d'images sans extension, classes en minuscules
[~, nomsImages] = cellfun(@fileparts, imdsTest.Files, UniformOutput=false);
nomsClasses     = cellstr(lower(string(predTest)));

% containers.Map : permet des clés qui ne sont pas des noms de champs valides
% (ex. "0001"), ce que ne permet pas une struct
dico = containers.Map(nomsImages, nomsClasses);
txtJson = jsonencode(dico, PrettyPrint=true);

fid = fopen("predictions.json", "w");
fwrite(fid, txtJson, "char");
fclose(fid);
fprintf("\npredictions.json généré (%d images).\n", numel(nomsImages));

%% ============================ FONCTIONS LOCALES ==========================
function couches = blocSeparable(nbFiltres, nom)
% Convolution séparable : depthwise 3x3 (1 filtre par canal) puis pointwise 1x1.
% Params : 9*Cin + Cin*Cout au lieu de 9*Cin*Cout pour une conv classique.
    couches = [
        groupedConvolution2dLayer(3, 1, "channel-wise", Padding="same", Name=nom+"_dw")
        batchNormalizationLayer(Name=nom+"_dw_bn")
        reluLayer(Name=nom+"_dw_relu")
        convolution2dLayer(1, nbFiltres, Name=nom+"_pw")
        batchNormalizationLayer(Name=nom+"_pw_bn")
        reluLayer(Name=nom+"_pw_relu")
    ];
end