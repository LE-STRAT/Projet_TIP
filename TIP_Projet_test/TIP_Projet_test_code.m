%% Food-11 : script de départ (transfert d'apprentissage, Matlab R2024a+)
% Prérequis : Deep Learning Toolbox + le support package du réseau choisi
% (ex. "Deep Learning Toolbox Model for MobileNet-v2 Network", via Add-Ons).
clear; clc;

%% 1. Données : adapter le chemin vers le dossier dézippé
trainDir = "C:\Users\HP\Desktop\INSA\Cours Master 1\COURS S1\Traitement d'images et Paroles\TIP_Projet_test\food-11\train";
testDir  = "C:\Users\HP\Desktop\INSA\Cours Master 1\COURS S1\Traitement d'images et Paroles\TIP_Projet_test\food-11\test";

imds = imageDatastore(trainDir, IncludeSubfolders=true, LabelSource="foldernames");
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");
classNames = categories(imds.Labels);
numClasses = numel(classNames);

%% 2. Réseau pré-entraîné : on remplace la dernière couche (11 classes)
% Candidats légers : "squeezenet", "mobilenetv2", "shufflenet", "resnet18"
netName = "resnet18";
net = imagePretrainedNetwork(netName, NumClasses=numClasses);
inputSize = [128 128 3];   % ex. [224 224 3]

% Nombre de paramètres (sert au score de frugalité)
nParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf("Paramètres : %.2f M\n", nParams/1e6);

% Redimensionnement seul (pas d'augmentation pour la version de base)
% Pour tester l'augmentation plus tard, ajouter dans augmentedImageDatastore :
%   DataAugmentation=imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15])
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ColorPreprocessing="gray2rgb");
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal, ColorPreprocessing="gray2rgb");

%% 3. Entraînement (à faire varier : learning rate, epochs, batch size...)
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
figure; confusionchart(YTrue, YPred);   % pour l'analyse dans le rapport

%% 5. Prédictions sur le test + JSON (à lancer seulement avec le meilleur modèle)
imdsTest = imageDatastore(testDir);
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest, ColorPreprocessing="gray2rgb");
YTest    = scores2label(minibatchpredict(net, augTest), classNames);

fid = fopen("predictions.json", "w");
fprintf(fid, "{\n");
n = numel(imdsTest.Files);
for i = 1:n
    [~, name] = fileparts(imdsTest.Files{i});
    sep = ","; if i == n, sep = ""; end
    fprintf(fid, '    "%s": "%s"%s\n', name, lower(string(YTest(i))), sep);
end
fprintf(fid, "}\n");
fclose(fid);