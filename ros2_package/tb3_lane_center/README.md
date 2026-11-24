# TurtleBot3 Burger — rester entre le bord jaune et le bord blanc

Le package tb3_lane_center utilise le réseau jaune du projet et un second petit U-Net appris sur les polygones white_line du même dataset. Les fichiers ONNX sont fournis. Le centre est estimé à deux distances en moyennant les abscisses des bords appariés sur les mêmes lignes de l'image. Le contrôleur utilise la position du milieu proche et le déplacement du milieu plus loin pour anticiper le virage. Il suppose une caméra centrée, orientée vers l'avant. Depuis la v0.2.0, le jaune peut être à gauche OU à droite du blanc (paramètre lane_side). Le point milieu en pixels est une approximation du milieu physique.

## Installer sur le PC Humble déjà utilisé pour tb3_yellow_dl

    source /opt/ros/humble/setup.bash
    mkdir -p ~/tb3_dl_ws/src
    unzip ~/Downloads/tb3_lane_center_humble.zip -d ~/tb3_dl_ws/src
    cd ~/tb3_dl_ws
    colcon build --symlink-install --packages-select tb3_lane_center
    source install/setup.bash
    ros2 pkg executables tb3_lane_center

Garder la base et le pilote CSI ouverts sur le Pi. Arrêter tous les anciens contrôleurs de mouvement sur le PC. Chaque terminal PC utilisant ce package doit charger ROS Humble et ~/tb3_dl_ws/install/setup.bash. Les dépendances d'exécution sont identiques à celles du projet U-Net jaune.

## Aperçu immobile

    ros2 run tb3_lane_center follow --ros-args -r image_raw:=/camera/image_raw

Depuis un autre terminal, lancer ros2 run rqt_image_view rqt_image_view. Sélectionner /lane_overlay. Rose = jaune, cyan = blanc. Deux points verts indiquent les milieux utilisés pour calculer la direction. Aucun point vert = commande nulle. Vérifier sur les portions droites ET en courbe que les deux points sont au milieu de la voie. Vérifier aussi quand un bord sort du champ. Si le ruban touche le bord d'une plateforme surélevée, faire l'essai au sol. Ce système ne détecte pas les chutes.

    ros2 topic echo /lane_cmd_vel

Avec la caméra alignée, sur une ligne droite bien centrée, angular.z doit être proche de zéro. Avec les deux centres décalés vers la gauche, la rotation doit être positive, et inversement. Si les images sont trop anciennes, le terminal signale le délai et les vitesses restent nulles. L'ancien target_x est remplacé par le milieu des deux bords.

## Premier essai au sol

Arrêter l'aperçu par Ctrl+C. Poser le robot sur une piste dégagée, au centre, dans le sens jaune à gauche. Garder l'interrupteur à portée de main.

    ros2 run tb3_lane_center follow --ros-args -r image_raw:=/camera/image_raw -r lane_cmd_vel:=/cmd_vel

Vitesse avant limitée à 0,035 m/s. Pour arrêter le suiveur depuis son terminal utiliser Ctrl+C. Si le robot continue de rouler, utiliser l'interrupteur. Le système ne prend pas en compte les obstacles. Les marquages ambigus, le montage de la caméra et les nouveaux éclairages peuvent encore entraîner des erreurs. Le contrôleur arrête le robot lorsque les deux bords ne donnent pas un couple cohérent dans les deux bandes ou que le délai est trop important.

## Reproduire sur le dataset du projet

train.py (PyTorch CPU 2.5.1, NumPy, SciPy, Pillow, ONNX et ONNX Runtime) entraîne le modèle blanc sur le ZIP d'origine avec les mêmes exclusions de doublons que pour le jaune. Fournir --out pour enregistrer un nouvel essai ailleurs. Les 183 images d'entraînement, 61 de validation et 14 de test proviennent des splits existants. evaluate_center.py compare sur le test le milieu calculé sur les masques du modèle au milieu calculé sur les annotations disponibles.

    python train.py /chemin/turtlebot3_autorace_dataset_to_fill.zip --out ./essai_blanc
    python evaluate_center.py /chemin/turtlebot3_autorace_dataset_to_fill.zip

Les métriques du modèle blanc livré et du calcul de centre sont dans models/metrics.json et models/center_test.json. La précision de masque est indépendante de la capacité à repérer les deux bords simultanément.

## v0.2.0 — sens de marche inversé (lane_side)

La v0.1.0 n'acceptait que le jaune à gauche du blanc : sur toutes les séquences *_inverse du dataset, le robot restait à l'arrêt (0 % des images). Le paramètre `lane_side` corrige cela :

- `auto` (défaut) : essaie les deux ordres et garde celui qui a le plus de lignes appariées.
- `yellow_left` : comportement identique à la v0.1.0 (vérifié image par image sur les 334 images).
- `yellow_right` : sens inverse de la piste.

    ros2 run tb3_lane_center follow --ros-args -r image_raw:=/camera/image_raw -p lane_side:=auto

Si vous connaissez le sens de marche, fixez `yellow_left` ou `yellow_right` : c'est plus robuste qu'`auto` quand d'autres rubans de la piste sont visibles.

Images du dataset complet (334) où le contrôleur produit une commande : 162 (48,5 %) en v0.1.0 → 302 (90,4 %) en v0.2.0. Les 32 images restantes montrent surtout un seul bord visible ou des bords lointains ; le robot s'arrête alors volontairement. Ce taux mesure la disponibilité d'une commande, pas sa justesse (le dataset fourni ne contient pas les annotations JSON nécessaires pour mesurer l'erreur de centre).

## v0.2.1 — max_camera_age

Nouveau paramètre `max_camera_age` (défaut 0.4 s, borne 0.05–1.0 s). D'abord synchroniser les horloges Pi/PC (chrony). Si le délai réel reste juste au-dessus de 0.4 s, `-p max_camera_age:=0.6` (à 0,035 m/s, 0,6 s = environ 2 cm parcourus sur une image ancienne).

## v0.2.2 — max_speed

Nouveau paramètre `max_speed` (défaut 0.035 m/s, maximum 0.12 m/s). Vitesse réduite automatiquement dans les virages. Augmenter par paliers : 0.05, puis 0.07, puis 0.09.

## v0.3.0 — mode récupération dans les virages

Quand les deux bords ne peuvent pas être appariés (virage serré, un bord sort de l'image), le robot ne s'arrête plus immédiatement : il vise une cible calculée à partir du bord encore visible (point orange dans /lane_overlay), à 40 % de max_speed, pendant au plus `recovery_time` secondes (défaut 1.5 s, 0 = désactivé) après la dernière détection des deux bords. Au-delà, ou si aucun bord n'est visible, il s'arrête.

## v0.3.1

Récupération plus forte : gain 3.0, rotation sur place si la correction est forte (|yaw| > 0.45). Le terminal affiche MODE: LANE / RECOVERY / STOP (raison) à chaque changement.

## v0.4.0 — conduite fluide à vitesse élevée

- Le braquage augmente avec max_speed au-delà de 0.05 m/s (même courbure de trajectoire), limité à 1.5 rad/s. Identique à avant pour max_speed <= 0.05.
- `hold_time` (0.3 s) : si la voie est perdue brièvement, garde la dernière commande LANE au lieu de passer en RECOVERY.
- `smoothing` (0.5) : lissage des commandes (0 = aucun, 0.9 = très lisse mais plus lent à réagir).
- RECOVERY sans rotation sur place : virage progressif, vitesse réduite.
- max_speed jusqu'à 0.22 m/s.
