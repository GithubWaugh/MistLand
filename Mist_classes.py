# MISTLAND : CLASSES
class monde :
    def __init__(self) :
        self.name = "Mistland"
        self.emprise = 25
        self.topographie = Topographie()

class Topographie :
    def __init__(self):
        self.cartes = []
    def generation(self, size: int = 10) :
        self.carte = [[Zone() for _ in range(size)] for _ in range(size)]
    def afficher(self) :
        for ligne in self.carte :
            for zone in ligne :
                print(f"Altitude : {zone.altitude}, Terrain : {zone.terrain.type}, Vegetation : {zone.vegetation.rase} et {zone.vegetation.haute}, Humidite : {zone.humidite}, Temperature : {zone.temperature}")
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
        self.types = {"roche":"0", "terre":"1", "eau":"2", "sable":"3", "glace":"4"}
        self.type = list(self.types.keys())[0]

class Vegetation :
    def __init__(self):
        self.rases = ["herbe", "fleurs", "lichen"]
        self.rase = self.rases[0]
        self.hautes = ["arbres", "buissons", "vignes"]
        self.haute = self.hautes[0]

