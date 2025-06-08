class GenerationInfo:
    waterThreshold = 0.505
    mountainThreshold = 0.5125

    tileSize = 8

    territorySize = 100

    territoryVisionRange = 20

    mapSizeScalar = 1.5


class ResourceInfo:
    resourceTypes = ['wood', 'stone', 'iron', 'pine', 'amber']
    numResources = len(resourceTypes)

    spawnRates = {'wood': 0.021,
                  'stone': 0.014,
                  'iron': 0.025,
                  'pine': 0.005,
                  'amber': 0.006}

    @staticmethod
    def getSpawnableTiles(resourceType, tiles):
        spawnableTiles = {'wood': [t for t in tiles if t.isLand and not t.isMountain and not t.isCoast and (len(t.adjacent) == 6)],
                          'stone': [t for t in tiles if t.isLand and not t.isCoast and (len(t.adjacent) == 6)],
                          'iron': [t for t in tiles if t.isLand and t.isMountain and not t.isCoast and (len(t.adjacent) == 6)],
                          'pine': [t for t in tiles if t.isLand and not t.isMountain and not t.isCoast and (len(t.adjacent) == 6)],
                          'amber': [t for t in tiles if t.isLand and t.isMountain and not t.isCoast and (len(t.adjacent) == 6)]}
        return spawnableTiles[resourceType]


class StructureInfo:
    harborSpawnRate = 0.02


class ShipInfo:
    shipTypes = ['fluyt', 'carrack', 'cutter', 'corsair', 'longShip', 'galleon']
    shipClasses = {'fluyt': 'TradeShip', 'carrack': 'TradeShip', 'cutter': 'Warship', 'corsair': 'Warship', 'longShip': 'LongShip', 'galleon': 'LongShip', }

    shipHPs = {'fluyt': 50, 'carrack': 100, 'cutter': 75, 'corsair': 90, 'longShip': 80, 'galleon': 120}
    shipMSs = {'fluyt': 0.3, 'carrack': 0.2, 'cutter': 0.4, 'corsair': 0.35, 'longShip': 0.25, 'galleon': 0.15}
    shipVisionRanges = {'fluyt': 24, 'carrack': 20, 'cutter': 16, 'corsair': 18, 'longShip': 22, 'galleon': 18}
    shipDMGs = {'fluyt': 0, 'carrack': 0, 'cutter': 15, 'corsair': 25, 'longShip': 10, 'galleon': 20}
    shipAttackRanges = {'fluyt': 0, 'carrack': 0, 'cutter': 3, 'corsair': 4, 'longShip': 2, 'galleon': 3}
    shipStorageCapacities = {'fluyt': 150, 'carrack': 200, 'cutter': 0, 'corsair': 0, 'longShip': 100, 'galleon': 250}
    shipSizes = {'fluyt': 35, 'carrack': 40, 'cutter': 30, 'corsair': 35, 'longShip': 35, 'galleon': 50}


class uiInfo:
    bottomUIBarSize = 0.07


class Cols:
    oceanBlue = [59, 95, 111]
    oceanGreen = [73, 120, 122]
    lightOceanGreen = [86, 142, 143]
    oceanFoam = [148, 182, 180]
    sandyBrown = [172, 146, 95]
    darkSandyBrown = [159, 143, 91]
    oliveGreen = [95, 115, 84]
    darkOliveGreen = [54, 64, 57]
    mountainBlue = (83, 78, 90)
    darkMountainBlue = [42, 40, 52]
    light = [220, 216, 201]
    dark = [19, 21, 22]
    crimson = [94, 32, 47]
    brightCrimson = [85, 42, 56]
    cloudLight = [176, 185, 205]
    cloudMedium = [124, 134, 156]
    cloudDark = [105, 116, 140]
    debugRed = [255, 96, 141]
