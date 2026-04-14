from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Product:
    avg_sales: int
    product_id: int
    weight: float
    length: int
    width: int
    depth: int

    @property
    def volume(self) -> int:
        return self.length * self.width * self.depth


@dataclass
class Box:
    x: float
    y: float
    z: float

    def remap_axis(self, packed: list[PackedProduct]) -> Box:
        """Remap axes so Z is height (largest-area face is bottom) and X is
        the longer horizontal axis.

        Mutates each *PackedProduct* in-place and returns a new remapped Box.
        """
        carton_bottom = self._carton_bottom()
        axis_perm = [(0, 1, 2), (1, 2, 0), (0, 2, 1)][carton_bottom]
        vals = (self.x, self.y, self.z)
        rx, ry, rz = vals[axis_perm[0]], vals[axis_perm[1]], vals[axis_perm[2]]
        if ry > rx:
            axis_perm = (axis_perm[1], axis_perm[0], axis_perm[2])
            rx, ry = ry, rx

        for p in packed:
            pos = (p.x, p.y, p.z)
            dim = (p.dx, p.dy, p.dz)
            p.x, p.y, p.z = pos[axis_perm[0]], pos[axis_perm[1]], pos[axis_perm[2]]
            p.dx, p.dy, p.dz = dim[axis_perm[0]], dim[axis_perm[1]], dim[axis_perm[2]]

        return Box(x=rx, y=ry, z=rz)

    def _carton_bottom(self) -> int:
        """Return the axis index (0=Z, 1=X, 2=Y) whose perpendicular face
        has the largest area."""
        faces = (self.x * self.y, self.y * self.z, self.x * self.z)
        return faces.index(max(faces))


@dataclass
class PackedProduct:
    """Solver output: a Product placed in the box with solved position and
    rotated dimensions."""

    product: Product
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
