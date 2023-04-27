class ECNumber:
    def __init__(self, number, coordinate, color, radius):
        self.number = number
        self.coordinate = coordinate
        self.color = color
        self.radius = radius

    def __str__(self):
        return (
            f"(EC: {self.number}, Coordinate: {self.coordinate}, Color: {self.color})"
        )
