class GenerationInfo:
    waterThreshold = 0.505
    mountainThreshold = 0.54

    tileSize = 6

    territorySize = 100


class ResourceInfo:
    resourceTypes = ['wood', 'stone', 'iron', 'pine', 'amber']
    numResources = len(resourceTypes)

    spawnRates = {'wood': 0.008,
                  'stone': 0.006,
                  'iron': 0.006,
                  'pine': 0.003,
                  'amber': 0.003}

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
    shipMSs = {'fluyt': 0.6, 'carrack': 0.4, 'cutter': 0.8, 'corsair': 0.7, 'longShip': 0.5, 'galleon': 0.3}
    shipVisionRanges = {'fluyt': 12, 'carrack': 10, 'cutter': 8, 'corsair': 9, 'longShip': 11, 'galleon': 9}
    shipDMGs = {'fluyt': 0, 'carrack': 0, 'cutter': 15, 'corsair': 25, 'longShip': 10, 'galleon': 20}
    shipAttackRanges = {'fluyt': 0, 'carrack': 0, 'cutter': 3, 'corsair': 4, 'longShip': 2, 'galleon': 3}
    shipStorageCapacities = {'fluyt': 150, 'carrack': 200, 'cutter': 0, 'corsair': 0, 'longShip': 100, 'galleon': 250}
    shipSizes = {'fluyt': 35, 'carrack': 40, 'cutter': 30, 'corsair': 35, 'longShip': 35, 'galleon': 50}
