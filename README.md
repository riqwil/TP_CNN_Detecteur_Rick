TP_CNN_Detecteur
L'objectif de ce TP est de vous permettre de vous familiariser avec l'utilisation de CNN et l'utilisation du transfer learning comme technique permettant d'éviter de réapprendre des modèles from scratch, et apprendre à manipuler des données image ou vidéo pour l'apprentissage de modèles.

Le but pour vous sera de télécharger un couple détecteur/classifieur déjà entrainés sur un premier dataset comme COCO, imageNET ou PascalVOC, d'identifier un nouveau cas d'usage qui vous intéresse, cela peut être :

Détection d'humain et de vahicules mais avec des vues CCTV Détection de plaques d'immatriculation Détection de différentes espèces de plantes

Le but est de partir d'un classifieur et d'un détecteur ayant appris sur un certain dataset, et de devoir changer les classes en sortie pour les adapter à un nouveau cas d'usage (transfer learning) et d'utiliser un dataset de détection que vous aurez trouvé pour finaliser l'apprentissage sur ce nouveau cas d'usage.

Exemple :

Vous partez d'un YoloV8 appris sur COCO Vous voulez garder toutes les classes d'humains et de véhicules Vous Téléchargez le dataset VIRAT qui a des vidéos de vues CCTV pour des humains et véhicules Vous adaptez le modèle YoloV8 pour la détection de ces classes et vous réalisez le réentrainement des dernières couches Vous observez les performances de détection (mAP avec différents seuils d'IoU, AP par classe, precision, recall), et en fonction décidez s'il est nécessaire de refaire l'apprentissage en réapprenant plus de couches dans le modèle

Il est conseillé de choisir parmi les familles : Yolo, Inception et RetinaNet, étant les plus simples à utiliser, mais vous pouvez choisir le modèle que vous voulez.


test modif conflit réglé

