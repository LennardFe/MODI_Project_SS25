import sqlite3
from scipy.integrate import trapezoid

def get_theta():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    data = cur.execute("""SELECT timestamp, z from gyro_data""").fetchall()
    conn.close()
    x_time = [float(a[0])*1.0e-9 for a in data]
    z = [a[1] for a in data]
    return trapezoid(z, x_time)
