import sqlite3
from scipy.integrate import trapezoid

def get_theta(database_name="MODI"):
    conn = sqlite3.connect(f'assets/{database_name}.db', check_same_thread=False)
    cur = conn.cursor()
    data = cur.execute("""SELECT timestamp, x FROM gyro_data""").fetchall()
    conn.close()
    x_time = [float(a[0]) * 1.0e-9 for a in data]
    x_value = [a[1] if abs(a[1]) > 0.75 else 0.0 for a in data]
    return trapezoid(x_value, x_time)
