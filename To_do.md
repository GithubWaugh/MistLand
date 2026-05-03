# INTERFACE #
## ENREGISTREMENT ##
Possibilité de sauvegarder plusieurs "parties" dans le même répertoire (ficher json et fichier npz avec un prefixe identique par exemple)
## AMELIORER LE FONCTIONNEMENT DES OVERLAYS ##
Touche "o" ou menu ouvre une fenêtre présentant la liste des overlays, en trois catégories :
- Base pour le fond de carte : 
    - couche en niveau de gris représentant la pente et l'orientation de chaque cellule : les cellules orientées nord-ouest sont éclairées, celle orientées Sud-Est dans une ombre relative, tout cela modulé par la pente (plus la pente est forte, plus le ton est foncé) (physiquement faux évidemment, mais similaire à la logique utilisée dans les cartes de randonnée): nécessite de calculer les gradients de pente
    - plage de luminosité .25 à .75 (éviter les tons extrêmes)
- Layer 1, au choix et de manière exclusive (Remplacement des composantes Hue et Saturation de la couche inférieure) :
    - types de sol (rock, sand, soil, water). 
    - température (dégradé existant)
    - pression
    - humidité aérienne
    - humidité au sol
- layer 2 : végétation et autres éléments au sol, au choix et de manière exclusive :
    - peut être neutre (désactivé)
    - végétation indiquée en surimpression :
        - lichen : quelques points vert-jaune pâle, opacité 0.5
        - grass : points verts, plus nombreux, opacité O.5
        - shrubs : points denses, verts, opacité variable de 0.5 à 1
        - trees : points, regroupés par deux, opaques, vert plus foncé
    - nutriments : points ocres, avec une densité dépendant de la quantité de nutriments présents
    - sédiments (pas encore implémentés, similaire aux nutriments) : points gris, densité selon quantité
- layer 3 : éléments atmosphériques : 
    - Peut être neutre (désactivé)
    - streamers du vent
        - sur le système déjà en place, ajouter une pointe de flèche minimaliste pour indiquer le sens du vent
    - pluie


# SIMULATION #
## SEDIMENTS ##
Ajouter une information supplémentaire sur chaque cellule : la quantité de sédiments.
Les sédiments ne sont pas une quantité statique sur la planète.
Lors de la génération, les cellules de type "soil" recoivent une quantité aléatoire de sédiments
### Génération des sédiments ###
La présence d'eau sur la roche nue (bare) génère des sédiments
### Transport des sédiments ###
Les sédiments sont partiellement emportés par le ruissellement vers les cellules concernées (altitude inférieure) où ils s'ajoutent aux sédiments déjà présents
### Destruction des sédiments ###
Les sédiments d'une cellule "Lac" (flooded) disparaissent
### Enrichissement du sable ###
Si une cellule de type "SAND" contient plus d'un certain seuil de sédiment, elle devient de type "SOIL"
Attention, cela implique que le type d'une cellule n'est plus statique, mais peut varier.
### Appauvrissement du sol ###
De façon inverse, si une cellule de type "SOIL" contient moins d'un certain seuil de sédiments, elle change de type pour "SAND".
Les plantes existantes sont conservées.


## PLUIE D'ALTITUDE#
Réflechir à un moyen de faire pleuvoir sur les montagnes -> Elevation de masse d'air entraine un refroidissement et une condensation, mécanisme qui n'existe pas encore dans le programme.

## BARE devient ROCK ##
Remplacer les références à 'bare' par le terme 'rock' 