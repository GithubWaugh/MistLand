# MISTLAND : CLASSES
import random


class monde :
    def __init__(self) :
        self.name = "Mistland"
        self.emprise = 25
        self.topographie = Topographie()

class Topographie :
    def __init__(self):
        self.cartes = []
    def generation_carte(self, size: int = 10) :
        self.carte = [[Zone() for _ in range(size)] for _ in range(size)]
        self.generation_altitude()
    def flouter_altitude(self) :
        for i in range(len(self.carte)) :
            for j in range(len(self.carte[i])) :
                self.carte[i][j].altitude = int((self.carte[(i-1)%len(self.carte)][j].altitude + self.carte[(i+1)%len(self.carte)][j].altitude + self.carte[i][(j-1)%len(self.carte[i])].altitude + self.carte[i][(j+1)%len(self.carte[i])].altitude) / 4)
    def generation_altitude(self) :
        for ligne in self.carte :
            for zone in ligne :
                zone.altitude = random.randint(0, 9)
    def afficher_terrain(self) :
        for ligne in self.carte :
            dessin_ligne = ""
            for zone in ligne :
                dessin_ligne += f"{zone.terrain.types[zone.terrain.type]} "
            print(dessin_ligne)
        print("\n")
    def afficher_altitude(self) :
        for ligne in self.carte :
            dessin_ligne = ""
            for zone in ligne :
                dessin_ligne += f"{zone.altitude} "
            print(dessin_ligne)
        print("\n")

class Zone :
    def __init__(self):
        self.altitude = 0
        self.terrain = Terrain()
        self.vegetation = Vegetation()
        self.humidite = 0
        self.temperature = 0

class Terrain :
    def __init__(self):
        self.types = {"roche":"#", "terre":" ", "eau":"~", "sable":".", "glace":"*"}
        self.type = list(self.types.keys())[0]

class Vegetation :
    def __init__(self):
        self.rases = ["herbe", "fleurs", "lichen"]
        self.rase = self.rases[0]
        self.hautes = ["arbres", "buissons", "vignes"]
        self.haute = self.hautes[0]

# Exemple d'utilisation
test = monde()
test.topographie.generation_carte(5)
test.topographie.afficher_altitude()
test.topographie.flouter_altitude()
test.topographie.afficher_altitude()
