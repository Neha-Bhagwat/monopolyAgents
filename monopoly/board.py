from dataclasses import dataclass
from enum import Enum, auto


class Kind(Enum):
    GO = auto()
    PROPERTY = auto()
    CHANCE = auto()
    CHEST = auto()
    TAX = auto()
    JAIL = auto()
    FREE_PARKING = auto()
    GO_TO_JAIL = auto()


@dataclass(frozen=True)
class Property:
    group: str
    price: int
    build_cost: int
    rents: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Square:
    name: str
    kind: Kind
    property: Property | None = None
    tax: int = 0


def street(
    name: str, group: str, price: int, build_cost: int,
    rents: tuple[int, int, int, int, int, int],
) -> Square:
    return Square(name, Kind.PROPERTY, Property(group, price, build_cost, rents))


BOARD = (
    Square("Go", Kind.GO),
    street("Mediterranean Avenue", "brown", 60, 50, (2, 10, 30, 90, 160, 250)),
    Square("Community Chest", Kind.CHEST),
    street("Baltic Avenue", "brown", 60, 50, (4, 20, 60, 180, 320, 450)),
    Square("Income Tax", Kind.TAX, tax=200),
    street("Oriental Avenue", "light blue", 100, 50, (6, 30, 90, 270, 400, 550)),
    Square("Jail", Kind.JAIL),
    street("Vermont Avenue", "light blue", 100, 50, (6, 30, 90, 270, 400, 550)),
    Square("Chance", Kind.CHANCE),
    street("Connecticut Avenue", "light blue", 120, 50, (8, 40, 100, 300, 450, 600)),
    street("St. Charles Place", "pink", 140, 100, (10, 50, 150, 450, 625, 750)),
    Square("Community Chest", Kind.CHEST),
    Square("Free Parking", Kind.FREE_PARKING),
    street("States Avenue", "pink", 140, 100, (10, 50, 150, 450, 625, 750)),
    street("Virginia Avenue", "pink", 160, 100, (12, 60, 180, 500, 700, 900)),
    Square("Chance", Kind.CHANCE),
    street("St. James Place", "orange", 180, 100, (14, 70, 200, 550, 750, 950)),
    street("Tennessee Avenue", "orange", 180, 100, (14, 70, 200, 550, 750, 950)),
    Square("Go to Jail", Kind.GO_TO_JAIL),
    street("New York Avenue", "orange", 200, 100, (16, 80, 220, 600, 800, 1000)),
    street("Kentucky Avenue", "red", 220, 150, (18, 90, 250, 700, 875, 1050)),
    Square("Luxury Tax", Kind.TAX, tax=100),
    street("Indiana Avenue", "red", 220, 150, (18, 90, 250, 700, 875, 1050)),
    street("Illinois Avenue", "red", 240, 150, (20, 100, 300, 750, 925, 1100)),
)

JAIL = 6
GROUPS = {
    group: tuple(i for i, square in enumerate(BOARD) if square.property and square.property.group == group)
    for group in ("brown", "light blue", "pink", "orange", "red")
}


class CardMove(Enum):
    ADVANCE_TO_GO = auto()
    GO_TO_JAIL = auto()


CHANCE = (CardMove.ADVANCE_TO_GO, CardMove.GO_TO_JAIL, 50, -50, 100, -25)
CHEST = (CardMove.ADVANCE_TO_GO, CardMove.GO_TO_JAIL, 100, -50, 25, 50)
