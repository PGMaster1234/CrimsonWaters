class GenerationInfo:
    waterThreshold = 0.51
    mountainThreshold = 0.54

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
        spawnableTiles = {'wood': [t for t in tiles if t.isLand and not t.isMountain],
                          'stone': [t for t in tiles if t.isLand and not t.isCoast],
                          'iron': [t for t in tiles if t.isLand and t.isMountain],
                          'pine': [t for t in tiles if t.isLand and not t.isMountain],
                          'amber': [t for t in tiles if t.isLand and t.isMountain]}
        return spawnableTiles[resourceType]


class StructureInfo:
    harborSpawnRate = 0.001
