# SEDIMENTS #
Ajouter une information supplémentaire sur chaque cellule : la quantité de sédiments.
Les sédiments ne sont pas une quantité statique sur la planète.
Lors de la génération, les cellules de type "soil" recoivent une quantité aléatoire de sédiments
## Génération des sédiments ##
La présence d'eau sur la roche nue (bare) génère des sédiments
## Transport des sédiments ##
Les sédiments sont partiellement emportés par le ruissellement vers les cellules concernées (altitude inférieure) où ils s'ajoutent aux sédiments déjà présents
## Destruction des sédiments ##
Les sédiments d'une cellule "Lac" (flooded) disparaissent
## Enrichissement du sable ##
Si une cellule de type "SAND" contient plus d'un certain seuil de sédiment, elle devient de type "SOIL"
Attention, cela implique que le type d'une cellule n'est plus statique, mais peut varier.
## Appauvrissement du sol ##
De façon inverse, si une cellule de type "SOIL" contient moins d'un certain seuil de sédiments, elle change de type pour "SAND".
Les plantes existantes sont conservées.


# PLUIE D'ALTITUDE
Réflechir à un moyen de faire pleuvoir sur les montagnes -> Elevation de masse d'air entraine un refroidissement et une condensation, mécanisme qui n'existe pas encore dans le programme.

# BARE devient ROCK
Remplacer les références à 'bare' par le terme 'rock' 