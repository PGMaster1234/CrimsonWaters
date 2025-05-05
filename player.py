from controlPanel import ShipInfo, ResourceInfo


class PlayerHandler:
    def __init__(self):
        self.players = []

    def add_player(self, player):
        self.players.append(player)


class Player:
    def __init__(self, ip, runtimePort, territory):
        self.ip = ip
        self.runtimePort = runtimePort
        self.territory = territory

        self.ships = []
        self.harbors = []
        self.resources = []

        self.resourceStorages = {r: 0 for r in ResourceInfo.resourceTypes}

    def spawnShip(self, shipType, homeHarbor):
        ship = ShipInfo.shipClasses[shipType](homeHarbor.tile, shipType, ShipInfo, ResourceInfo)
        self.ships.append(ship)
        return ship
