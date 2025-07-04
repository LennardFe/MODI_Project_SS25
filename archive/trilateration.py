import numpy as np


def trilaterate_3d_three_spheres(p1, r1, p2, r2, p3, r3):
    # Konvertiere Punkte in numpy-Arrays
    P1, P2, P3 = np.array(p1), np.array(p2), np.array(p3)

    # Basisvektoren des lokalen Koordinatensystems
    ex = P2 - P1
    ex = ex / np.linalg.norm(ex)
    i = np.dot(ex, P3 - P1)
    ey = P3 - P1 - i * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)

    # Koordinaten des Punktes in lokalem System
    x = (r1**2 - r2**2 + d**2) / (2 * d)
    y = ((r1**2 - r3**2 + i**2 + j**2) / (2 * j)) - ((i / j) * x)
    z_sq = r1**2 - x**2 - y**2

    if z_sq < 0:
        raise ValueError("Keine reale Lösung – die Kugeln schneiden sich nicht.")

    z = np.sqrt(z_sq)

    # Zwei mögliche Lösungen:
    result1 = P1 + x * ex + y * ey + z * ez
    result2 = P1 + x * ex + y * ey - z * ez

    return result1, result2


p1 = (0, 0, 0)
p2 = (-2, 0, 0)
p3 = (-2, 4, 0)
r1 = 5
r2 = 5.4
r3 = 2.1

sol1, sol2 = trilaterate_3d_three_spheres(p1, r1, p2, r2, p3, r3)
print("Lösung 1:", sol1)
print("Lösung 2:", sol2)
