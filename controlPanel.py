class GenerationInfo:
    waterThreshold = 0.505
    mountainThreshold = 0.5125

    tileSize = 25

    territorySize = 250


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
        spawnableTiles = {'wood': [t for t in tiles if t.isLand and not t.isMountain and not t.isCoast],
                          'stone': [t for t in tiles if t.isLand and not t.isCoast],
                          'iron': [t for t in tiles if t.isLand and t.isMountain and not t.isCoast],
                          'pine': [t for t in tiles if t.isLand and not t.isMountain and not t.isCoast],
                          'amber': [t for t in tiles if t.isLand and t.isMountain and not t.isCoast]}
        return spawnableTiles[resourceType]


class StructureInfo:
    harborSpawnRate = 0.02


class ShipInfo:
    shipTypes = ['fluyt', 'carrack', 'cutter', 'corsair', 'longShip', 'galleon']
    shipClasses = {'fluyt': 'TradeShip', 'carrack': 'TradeShip', 'cutter': 'Warship', 'corsair': 'Warship', 'longShip': 'LongShip', 'galleon': 'LongShip', }

    shipHPs = {'fluyt': 50, 'carrack': 100, 'cutter': 75, 'corsair': 90, 'longShip': 80, 'galleon': 120}
    shipMSs = {'fluyt': 0.3, 'carrack': 0.2, 'cutter': 0.4, 'corsair': 0.35, 'longShip': 0.25, 'galleon': 0.15}
    shipVisionRanges = {'fluyt': 12, 'carrack': 10, 'cutter': 8, 'corsair': 9, 'longShip': 11, 'galleon': 9}
    shipDMGs = {'fluyt': 0, 'carrack': 0, 'cutter': 15, 'corsair': 25, 'longShip': 10, 'galleon': 20}
    shipAttackRanges = {'fluyt': 0, 'carrack': 0, 'cutter': 3, 'corsair': 4, 'longShip': 2, 'galleon': 3}
    shipStorageCapacities = {'fluyt': 150, 'carrack': 200, 'cutter': 0, 'corsair': 0, 'longShip': 100, 'galleon': 250}
    shipSizes = {'fluyt': 35, 'carrack': 40, 'cutter': 30, 'corsair': 35, 'longShip': 35, 'galleon': 50}
