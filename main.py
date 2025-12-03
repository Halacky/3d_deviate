import numpy as np
import struct
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from scipy.ndimage import label

def generate_rounded_cube_point_cloud(filename='cube_half.ply', 
                                      size=10.0, 
                                      bend_depth=3.0,
                                      points_per_edge=80,
                                      hole_radius=2.0):
    """
    Генерирует PLY файл с облаком точек - красное полотно с неровностями,
    отверстием в центре и загнутыми вниз краями.
    """
    points = []
    colors = []
    
    half_size = size / 2.0
    
    def add_noise_to_surface(x, y, base_z):
        """Добавляет неровности к поверхности"""
        # Проверка отверстия в центре
        dist_from_center = np.sqrt(x**2 + y**2)
        if dist_from_center < hole_radius:
            return None  # Не добавляем точки в отверстии
        
        # Комбинация различных типов неровностей
        sharp_peaks = 0.5 * np.sin(x * 2) * np.cos(y * 2)
        smooth_waves = 0.3 * np.sin(x * 0.5) * np.sin(y * 0.5)
        random_noise = 0.2 * np.sin(x * 3.7 + y * 2.3) * np.cos(x * 1.8 - y * 4.1)
        radial = 0.15 * np.sin(dist_from_center * 1.5)
        
        total_displacement = sharp_peaks + smooth_waves + random_noise + radial
        
        return base_z + total_displacement
    
    def calculate_bend(x, y):
        """Вычисляет загиб краёв вниз"""
        dist_to_edge_x = half_size - abs(x)
        dist_to_edge_y = half_size - abs(y)
        dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
        
        bend_zone = half_size * 0.3
        
        if dist_to_edge < bend_zone:
            t = 1.0 - (dist_to_edge / bend_zone)
            bend_amount = bend_depth * (1 - np.cos(t * np.pi / 2))
            return -bend_amount
        
        return 0.0
    
    def get_color_gradient(x, y, z):
        """Создает красный градиент с вариациями"""
        base_r = 220
        base_g = 50
        base_b = 50
        
        variation = int(20 * np.sin(x * 0.5) * np.cos(y * 0.5))
        
        r = min(255, max(0, base_r + variation))
        g = min(255, max(0, base_g + variation // 2))
        b = min(255, max(0, base_b + variation // 2))
        
        return (r, g, b)
    
    # Генерация точек для верхней поверхности с загнутыми краями
    step = size / points_per_edge
    
    for i in range(points_per_edge + 1):
        for j in range(points_per_edge + 1):
            x = -half_size + i * step
            y = -half_size + j * step
            
            base_z = 0.0
            bend_z = calculate_bend(x, y)
            final_z = add_noise_to_surface(x, y, base_z + bend_z)
            
            if final_z is not None:
                points.append([x, y, final_z])
                colors.append(get_color_gradient(x, y, final_z))
    
    # Добавляем дополнительные точки для плавности на изгибах
    fine_step = step / 2
    bend_zone = half_size * 0.3
    
    for i in range(points_per_edge * 2 + 1):
        for j in range(points_per_edge * 2 + 1):
            x = -half_size + i * fine_step
            y = -half_size + j * fine_step
            
            dist_to_edge_x = half_size - abs(x)
            dist_to_edge_y = half_size - abs(y)
            dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
            
            if dist_to_edge < bend_zone:
                base_z = 0.0
                bend_z = calculate_bend(x, y)
                final_z = add_noise_to_surface(x, y, base_z + bend_z)
                
                if final_z is not None:
                    points.append([x, y, final_z])
                    colors.append(get_color_gradient(x, y, final_z))
    
    # Сохранение в PLY формат
    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)
    
    with open(filename, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Generated cloth surface with hole and bent edges
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        for point, color in zip(points, colors):
            f.write(struct.pack('fff', point[0], point[1], point[2]))
            f.write(struct.pack('BBB', color[0], color[1], color[2]))
    
    print(f"PLY файл '{filename}' успешно создан!")
    print(f"Количество точек: {len(points)}")
    
    return filename, points


def fit_plane_to_points(points):
    """
    Находит плоскость z = m x + n y + p,
    которая НЕ пересекает облако точек (лежит снизу) и
    минимизирует максимальное отклонение (chebyshev fit).

    Решается задача:
        minimize t
        s.t.  zi - (m xi + n yi + p) >= 0  для всех i  (точки не ниже плоскости)
              zi - (m xi + n yi + p) <= t  для всех i  (ограничиваем расстояние сверху)
              t >= 0
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    n_points = len(points)

    # Переменные: [m, n, p, t]
    # Начальное приближение: используем обычную МНК-плоскость, t = 1.0
    A = np.c_[X, Y, np.ones(n_points)]
    m0, n0, p0 = np.linalg.lstsq(A, Z, rcond=None)[0]
    x0 = np.array([m0, n0, p0, 1.0])

    def obj(x):
        # минимизируем t
        return x[3]

    def obj_grad(x):
        # градиент по [m, n, p, t]
        return np.array([0.0, 0.0, 0.0, 1.0])

    cons = []

    # Условие: zi - (m xi + n yi + p) >= 0
    def make_g_i(i):
        xi, yi, zi = X[i], Y[i], Z[i]
        def g_i(x):
            m, n, p, t = x
            return zi - (m*xi + n*yi + p)
        def g_i_jac(x):
            # d/d[m, n, p, t] (zi - m xi - n yi - p) = [-xi, -yi, -1, 0]
            return np.array([-xi, -yi, -1.0, 0.0])
        return {'type': 'ineq', 'fun': g_i, 'jac': g_i_jac}

    # Условие: t - (zi - (m xi + n yi + p)) >= 0  <=>  zi - (m xi + n yi + p) <= t
    # то есть g2_i(x) = t - (zi - (m xi + n yi + p)) >= 0
    def make_h_i(i):
        xi, yi, zi = X[i], Y[i], Z[i]
        def h_i(x):
            m, n, p, t = x
            return t - (zi - (m*xi + n*yi + p))
        def h_i_jac(x):
            # d/d[m, n, p, t] (t - zi + m xi + n yi + p) = [xi, yi, 1, 1]
            return np.array([xi, yi, 1.0, 1.0])
        return {'type': 'ineq', 'fun': h_i, 'jac': h_i_jac}

    for i in range(n_points):
        cons.append(make_g_i(i))
        cons.append(make_h_i(i))

    # t >= 0
    def g_t(x):
        return x[3]
    def g_t_jac(x):
        return np.array([0.0, 0.0, 0.0, 1.0])
    cons.append({'type': 'ineq', 'fun': g_t, 'jac': g_t_jac})

    res = minimize(
        obj, x0,
        method='SLSQP',
        jac=obj_grad,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
    )

    if not res.success:
        print("ВНИМАНИЕ: оптимизация не сошлась, используем МНК-плоскость.")
        m, n, p = m0, n0, p0
    else:
        m, n, p, t = res.x
        print(f"\nОптимизация завершена. t (макс. отклонение): {t:.4f}")

    print(f"\nПараметры плоскости (без пересечения облака):")
    print(f"z = {m:.4f}*x + {n:.4f}*y + {p:.4f}")
    
    return m, n, p


def calculate_perpendicular_distances(points, plane_params):
    """
    Вычисляет расстояния от каждой точки до плоскости по перпендикуляру.
    Для плоскости z = mx + ny + p, расстояние от точки (x0, y0, z0):
    d = (z0 - (m x0 + n y0 + p)) / sqrt(1 + m^2 + n^2)

    Здесь предполагается, что плоскость находится ниже облака:
    d >= 0 (или очень малые отрицательные из-за численных погрешностей).
    Отрицательные значения обрезаем до 0.
    """
    m, n, p = plane_params
    
    z_plane = m * points[:, 0] + n * points[:, 1] + p
    distances_raw = (points[:, 2] - z_plane) / np.sqrt(1 + m**2 + n**2)

    # Чистим возможные отрицательные значения (численные артефакты)
    distances = np.maximum(distances_raw, 0.0)
    
    print(f"\nСтатистика отклонений (неотрицательные):")
    print(f"Минимальное: {np.min(distances):.4f}")
    print(f"Максимальное: {np.max(distances):.4f}")
    print(f"Среднее: {np.mean(distances):.4f}")
    print(f"Стандартное отклонение: {np.std(distances):.4f}")
    
    return distances

def create_relief_map(points, distances, resolution=200, hole_radius=2.0):
    """
    Создает карту рельефа - проекцию отклонений на плоскость XY.
    Все отклонения считаются >= 0, цветовая шкала построена
    по положительной части ошибки (от 0 до max_dist).

    hole_radius — радиус отверстия в модели (должен совпадать с generate_rounded_cube_point_cloud).
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Создаем сетку
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Интерполируем расстояния на сетку
    from scipy.interpolate import griddata
    Z = griddata(points[:, :2], distances, (X, Y), method='linear')
    
    # --- НОВОЕ: вырезаем отверстие ---
    # Предполагаем, что отверстие по центру (0,0), как в генераторе
    R = np.sqrt(X**2 + Y**2)
    hole_mask = (R < hole_radius)
    # там, где отверстие, не хотим никаких значений
    Z[hole_mask] = np.nan
    # -------------------------------

    # Статистика по Z для цветовой шкалы
    # Важно: игнорируем NaN
    min_dist = np.nanmin(Z)
    max_dist = np.nanmax(Z)
    if np.isnan(min_dist) or np.isnan(max_dist):
        min_dist, max_dist = 0.0, 1.0  # fallback
    
    # Создаем цветовую карту только для положительных значений:
    # от 0 (светло-синий) до max_dist (ярко-красный)
    colors_map = [
        '#0000ff',  # 0   - синий
        '#0044ff',
        '#0088ff',
        '#00ccff',
        '#00ffff',  # ближе к середине
        '#88ff88',
        '#ffff00',
        '#ff8800',
        '#ff0000'   # max_dist - красный
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('relief_positive', colors_map, N=n_bins)
    
    plt.figure(figsize=(12, 10))
    
    im = plt.imshow(
        Z,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        cmap=cmap,
        aspect='equal',
        vmin=0.0,
        vmax=max_dist
    )
    
    cbar = plt.colorbar(im, label='Отклонение от плоскости (>= 0)')
    cbar.set_label('Отклонение от плоскости (>= 0)', fontsize=10)
    
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.title('Карта рельефа (положительные отклонения от опорной плоскости)')
    plt.grid(True, alpha=0.3)
    
    # Контурные линии (используем Z с NaN - matplotlib их автоматически пропустит)
    if not np.isnan(Z).all():
        levels = np.linspace(0.0, max_dist, 15)
        contours = plt.contour(X, Y, Z, levels=levels, colors='black', 
                               alpha=0.3, linewidths=0.5)
        plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('relief_map.png', dpi=300, bbox_inches='tight')
    print(f"\nКарта рельефа сохранена в 'relief_map.png'")
    
    return X, Y, Z

from matplotlib.colors import ListedColormap

def visualize_clusters_2d(X, Y, Z, labeled_Z, filename_prefix="clusters_projection"):
    """
    Создаёт 2D-визуализации кластеров:
      1) карта отклонений Z с контуром кластеров
      2) чистая карта кластеров (каждый кластер своим цветом)
    """
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # --- 1. Карта отклонений + контуры кластеров ---
    plt.figure(figsize=(12, 10))
    # фон – та же карта отклонений, что и в create_relief_map
    im = plt.imshow(
        Z,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(im, label='Отклонение от плоскости (>= 0)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Карта отклонений с контурами кластеров (Z > threshold)")

    # контуры по labeled_Z (где label > 0)
    mask_clusters = (labeled_Z > 0)
    # рисуем края кластеров
    plt.contour(
        X, Y, mask_clusters.astype(int),
        levels=[0.5],
        colors='red',
        linewidths=1.0
    )

    plt.tight_layout()
    out1 = f"{filename_prefix}_on_relief.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D-карта кластеров поверх рельефа сохранена в '{out1}'")

    # --- 2. Чистая карта кластеров ---
    plt.figure(figsize=(12, 10))

    labels = labeled_Z.copy()
    labels[labels < 0] = 0  # на всякий случай

    # Подготовим цветовую карту: 0 - фон (белый), 1..N - разные цвета
    n_clusters = labels.max()
    base_colors = ["#ffffff"]  # 0 - белый
    # несколько произвольных, но различимых цветов
    cluster_palette = [
        "#ff0000", "#00ff00", "#0000ff", "#ffff00",
        "#ff00ff", "#00ffff", "#ffa500", "#800080",
        "#008000", "#000080"
    ]
    # если кластеров больше 10 - будем циклически повторять palette
    for i in range(1, n_clusters + 1):
        base_colors.append(cluster_palette[(i - 1) % len(cluster_palette)])

    cmap_clusters = ListedColormap(base_colors)

    im2 = plt.imshow(
        labels,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='equal',
        cmap=cmap_clusters,
        vmin=0,
        vmax=max(1, n_clusters)
    )
    cbar = plt.colorbar(im2, ticks=range(0, n_clusters + 1))
    cbar.ax.set_yticklabels([f"фон (0)"] + [f"кластер {i}" for i in range(1, n_clusters + 1)])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Карта кластеров (каждый кластер отдельным цветом)")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    out2 = f"{filename_prefix}_labels.png"
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D-карта кластеров сохранена в '{out2}'")


def create_html_visualization(
        ply_filename, points,
        plane_params_bottom, distances_bottom,
        plane_params_top=None, distances_top=None,
        cluster_ids_peaks=None, clusters_info_peaks=None,
        cluster_ids_pits=None, clusters_info_pits=None,
        html_filename='visualization_with_plane.html'):
    """
    Визуализация:
      - нижняя плоскость (пики)
      - опционально верхняя плоскость (впадины)
      - точки кластеров пиков подсвечиваются жёлтым
      - точки кластеров впадин подсвечиваются голубым
    """
    import json

    m_bot, n_bot, p_bot = plane_params_bottom
    has_top = plane_params_top is not None
    if has_top:
        m_top, n_top, p_top = plane_params_top

    # Плоскость(и) для визуализации (берём диапазон по всем точкам)
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
    margin = 2.0

    def plane_corners(m, n, p):
        return [
            [x_min - margin, y_min - margin, float(m * (x_min - margin) + n * (y_min - margin) + p)],
            [x_max + margin, y_min - margin, float(m * (x_max + margin) + n * (y_min - margin) + p)],
            [x_max + margin, y_max + margin, float(m * (x_max + margin) + n * (y_max + margin) + p)],
            [x_min - margin, y_max + margin, float(m * (x_min - margin) + n * (y_max + margin) + p)],
        ]

    plane_points_bottom = plane_corners(m_bot, n_bot, p_bot)
    plane_points_top = plane_corners(m_top, n_top, p_top) if has_top else []

    # Читаем PLY
    with open(ply_filename, 'rb') as f:
        line = f.readline()
        num_vertices = 0
        while line.strip() != b'end_header':
            if line.startswith(b'element vertex'):
                num_vertices = int(line.split()[2])
            line = f.readline()
        
        points_data = []
        for _ in range(num_vertices):
            x, y, z = struct.unpack('fff', f.read(12))
            r, g, b = struct.unpack('BBB', f.read(3))
            points_data.append({
                'x': float(x), 'y': float(y), 'z': float(z),
                'r': float(r/255), 'g': float(g/255), 'b': float(b/255)
            })

    # Добавляем расстояния и 2 типа кластерных меток
    for i in range(len(points_data)):
        if i < len(distances_bottom):
            points_data[i]['dist_bottom'] = float(max(distances_bottom[i], 0.0))
        else:
            points_data[i]['dist_bottom'] = 0.0

        if distances_top is not None and i < len(distances_top):
            points_data[i]['dist_top'] = float(max(distances_top[i], 0.0))
        else:
            points_data[i]['dist_top'] = 0.0

        # пики
        if cluster_ids_peaks is not None and i < len(cluster_ids_peaks) and cluster_ids_peaks[i] >= 0:
            points_data[i]['is_peak'] = 1
        else:
            points_data[i]['is_peak'] = 0

        # впадины
        if cluster_ids_pits is not None and i < len(cluster_ids_pits) and cluster_ids_pits[i] >= 0:
            points_data[i]['is_pit'] = 1
        else:
            points_data[i]['is_pit'] = 0

    points_json = json.dumps(points_data, ensure_ascii=False)
    plane_bottom_json = json.dumps(plane_points_bottom, ensure_ascii=False)
    plane_top_json = json.dumps(plane_points_top, ensure_ascii=False)
    clusters_info_peaks_json = json.dumps(clusters_info_peaks if clusters_info_peaks is not None else [], ensure_ascii=False)
    clusters_info_pits_json = json.dumps(clusters_info_pits if clusters_info_pits is not None else [], ensure_ascii=False)

    min_dist_bot = float(np.min(distances_bottom))
    max_dist_bot = float(np.max(distances_bottom))
    mean_dist_bot = float(np.mean(distances_bottom))

    if distances_top is not None:
        min_dist_top = float(np.min(distances_top))
        max_dist_top = float(np.max(distances_top))
        mean_dist_top = float(np.mean(distances_top))
    else:
        min_dist_top = max_dist_top = mean_dist_top = 0.0

    # Подготовим HTML‑фрагменты, зависящие от has_top, отдельно
    if has_top:
        upper_plane_info_html = f"""
    <p><strong>Верхняя плоскость (впадины):</strong><br>
    z = {m_top:.4f}×x + {n_top:.4f}×y + {p_top:.4f}<br>
    Мин: {min_dist_top:.3f}, Макс: {max_dist_top:.3f}, Ср: {mean_dist_top:.3f}</p>
"""
        upper_plane_button_html = """
    <button class="toggle-btn" onclick="togglePlaneTop()">Показать/Скрыть верхнюю плоскость</button><br>
"""
    else:
        upper_plane_info_html = ""
        upper_plane_button_html = ""

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Визуализация с анализом рельефа (пики и впадины)</title>
<style>
body {{
    margin: 0;
    overflow: hidden;
    font-family: Arial, sans-serif;
}}
#container {{
    width: 100vw;
    height: 100vh;
}}
#info {{
    position: absolute;
    top: 10px;
    left: 10px;
    color: white;
    background: rgba(0,0,0,0.85);
    padding: 15px;
    border-radius: 5px;
    font-size: 13px;
    max-width: 360px;
}}
#controls {{
    position: absolute;
    bottom: 10px;
    left: 10px;
    color: white;
    background: rgba(0,0,0,0.85);
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
}}
.toggle-btn {{
    margin: 5px 0;
    padding: 5px 10px;
    background: #4CAF50;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
}}
.toggle-btn:hover {{
    background: #45a049;
}}
.cluster-badge {{
    display: inline-block;
    width: 10px;
    height: 10px;
    margin-right: 4px;
}}
</style>
</head>
<body>
<div id="container"></div>

<div id="info">
    <h3>Анализ рельефа модели</h3>
    <p><strong>Точек:</strong> {{pointsCount}}</p>
    <p><strong>Нижняя плоскость (пики):</strong><br>
    z = {m_bot:.4f}×x + {n_bot:.4f}×y + {p_bot:.4f}<br>
    Мин: {min_dist_bot:.3f}, Макс: {max_dist_bot:.3f}, Ср: {mean_dist_bot:.3f}</p>
{upper_plane_info_html}
    <p><strong>Кластеры пиков:</strong><br>
        <span id="clusters-peaks-summary"></span>
    </p>
    <p><strong>Кластеры впадин:</strong><br>
        <span id="clusters-pits-summary"></span>
    </p>
    <button class="toggle-btn" onclick="togglePlaneBottom()">Показать/Скрыть нижнюю плоскость</button><br>
{upper_plane_button_html}
    <button class="toggle-btn" onclick="togglePerpendiculars()">Показать/Скрыть перпендикуляры к нижней плоскости</button><br>
    <button class="toggle-btn" onclick="toggleClustersPeaks()">Пики: Подсветить/Скрыть</button><br>
    <button class="toggle-btn" onclick="toggleClustersPits()">Впадины: Подсветить/Скрыть</button>
</div>

<div id="controls">
    <strong>Управление:</strong><br>
    Левая кнопка — вращение<br>
    Колесо — масштаб<br>
    Правая кнопка — сдвиг
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
const pointsData = {points_json};
const planeBottom = {plane_bottom_json};
const planeTop = {plane_top_json};
const clustersInfoPeaks = {clusters_info_peaks_json};
const clustersInfoPits = {clusters_info_pits_json};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(15, 15, 15);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('container').appendChild(renderer.domElement);

// === Облако точек ===
const geometry = new THREE.BufferGeometry();
const positions = [];
const baseColors = [];
const peakMask = [];
const pitMask = [];

pointsData.forEach(p => {{
    positions.push(p.x, p.y, p.z);
    baseColors.push(p.r, p.g, p.b);
    peakMask.push(p.is_peak ? 1.0 : 0.0);
    pitMask.push(p.is_pit ? 1.0 : 0.0);
}});

geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.Float32BufferAttribute(baseColors, 3));
geometry.setAttribute('peakMask', new THREE.Float32BufferAttribute(peakMask, 1));
geometry.setAttribute('pitMask', new THREE.Float32BufferAttribute(pitMask, 1));

let showPeaks = true;
let showPits = true;

const pointsMaterial = new THREE.ShaderMaterial({{
    transparent: true,
    depthTest: true,
    uniforms: {{
        uShowPeaks: {{ value: 1.0 }},
        uShowPits: {{ value: 1.0 }},
    }},
    vertexShader: `
        attribute float peakMask;
        attribute float pitMask;
        varying vec3 vColor;
        varying float vPeakMask;
        varying float vPitMask;

        void main() {{
            vColor = color;
            vPeakMask = peakMask;
            vPitMask = pitMask;
            gl_PointSize = 3.5;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }}
    `,
    fragmentShader: `
        varying vec3 vColor;
        varying float vPeakMask;
        varying float vPitMask;
        uniform float uShowPeaks;
        uniform float uShowPits;

        void main() {{
            vec3 col = vColor;
            if (uShowPeaks > 0.5 && vPeakMask > 0.5) {{
                // пики - жёлтый
                col = vec3(1.0, 1.0, 0.0);
            }}
            if (uShowPits > 0.5 && vPitMask > 0.5) {{
                // впадины - голубой
                col = vec3(0.0, 1.0, 1.0);
            }}
            gl_FragColor = vec4(col, 1.0);
        }}
    `,
    vertexColors: true
}});

const points = new THREE.Points(geometry, pointsMaterial);
scene.add(points);

// === Плоскости ===
function makePlaneMesh(planePts, colorHex) {{
    const planeGeometry = new THREE.BufferGeometry();
    const planeVerts = new Float32Array([
        planePts[0][0], planePts[0][1], planePts[0][2],
        planePts[1][0], planePts[1][1], planePts[1][2],
        planePts[2][0], planePts[2][1], planePts[2][2],
        planePts[0][0], planePts[0][1], planePts[0][2],
        planePts[2][0], planePts[2][1], planePts[2][2],
        planePts[3][0], planePts[3][1], planePts[3][2]
    ]);
    planeGeometry.setAttribute('position', new THREE.BufferAttribute(planeVerts, 3));
    return new THREE.Mesh(
        planeGeometry,
        new THREE.MeshBasicMaterial({{color: colorHex, transparent: true, opacity: 0.3, side: THREE.DoubleSide}})
    );
}}

const planeMeshBottom = makePlaneMesh(planeBottom, 0x00ff00);
scene.add(planeMeshBottom);

let planeMeshTop = null;
if (planeTop.length === 4) {{
    planeMeshTop = makePlaneMesh(planeTop, 0xff0000);
    scene.add(planeMeshTop);
}}

// === Перпендикуляры к нижней плоскости ===
const perpendicularsGroup = new THREE.Group();
const mjs_bot = {plane_params_bottom[0]};
const njs_bot = {plane_params_bottom[1]};
const pjs_bot = {plane_params_bottom[2]};

pointsData.forEach((pt, idx) => {{
    if (idx % 50 !== 0) return;
    const z_plane = mjs_bot * pt.x + njs_bot * pt.y + pjs_bot;
    const lineGeom = new THREE.BufferGeometry();
    const verts = new Float32Array([
        pt.x, pt.y, pt.z,
        pt.x, pt.y, z_plane
    ]);
    lineGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    const lineMat = new THREE.LineBasicMaterial({{color: 0xffff00, opacity: 0.5, transparent: true}});
    const line = new THREE.Line(lineGeom, lineMat);
    perpendicularsGroup.add(line);
}});
scene.add(perpendicularsGroup);
perpendicularsGroup.visible = false;

// === Свет ===
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dl = new THREE.DirectionalLight(0xffffff, 0.5);
dl.position.set(10, 10, 10);
scene.add(dl);

// === Тогглы ===
function togglePlaneBottom() {{
    planeMeshBottom.visible = !planeMeshBottom.visible;
}}
function togglePlaneTop() {{
    if (planeMeshTop) planeMeshTop.visible = !planeMeshTop.visible;
}}
function togglePerpendiculars() {{
    perpendicularsGroup.visible = !perpendicularsGroup.visible;
}}
function toggleClustersPeaks() {{
    showPeaks = !showPeaks;
    pointsMaterial.uniforms.uShowPeaks.value = showPeaks ? 1.0 : 0.0;
}}
function toggleClustersPits() {{
    showPits = !showPits;
    pointsMaterial.uniforms.uShowPits.value = showPits ? 1.0 : 0.0;
}}

// === Мышь ===
let md = false, btn = -1, mx = 0, my = 0;
document.addEventListener('mousedown', e => {{ md = true; btn = e.button; mx = e.clientX; my = e.clientY; }});
document.addEventListener('mouseup', () => md = false);
document.addEventListener('contextmenu', e => e.preventDefault());

document.addEventListener('mousemove', e => {{
    if (!md) return;
    const dx = e.clientX - mx;
    const dy = e.clientY - my;
    if (btn === 0) {{
        points.rotation.y += dx * 0.01;
        points.rotation.x += dy * 0.01;
        planeMeshBottom.rotation.copy(points.rotation);
        if (planeMeshTop) planeMeshTop.rotation.copy(points.rotation);
        perpendicularsGroup.rotation.copy(points.rotation);
    }} else if (btn === 2) {{
        camera.position.x -= dx * 0.05;
        camera.position.y += dy * 0.05;
    }}
    mx = e.clientX; my = e.clientY;
}});

document.addEventListener('wheel', e => {{
    camera.position.z += e.deltaY * 0.01;
    camera.position.z = Math.max(5, Math.min(50, camera.position.z));
}});

// === Краткая инфа по кластерам ===
const peaksEl = document.getElementById('clusters-peaks-summary');
if (clustersInfoPeaks.length === 0) {{
    peaksEl.textContent = 'нет кластеров пиков выше порога';
}} else {{
    let html = '';
    clustersInfoPeaks.forEach(c => {{
        html += '<div>' +
                '<span class="cluster-badge" style="background:#ffff00;"></span>' +
                'ID ' + c.label + ': size=' + c.size +
                ', max=' + c.max_height.toFixed(2) +
                ', center=(' + c.center_x.toFixed(2) + ', ' + c.center_y.toFixed(2) + ')' +
                '</div>';
    }});
    peaksEl.innerHTML = html;
}}

const pitsEl = document.getElementById('clusters-pits-summary');
if (clustersInfoPits.length === 0) {{
    pitsEl.textContent = 'нет кластеров впадин выше порога';
}} else {{
    let html = '';
    clustersInfoPits.forEach(c => {{
        html += '<div>' +
                '<span class="cluster-badge" style="background:#00ffff;"></span>' +
                'ID ' + c.label + ': size=' + c.size +
                ', max=' + c.max_height.toFixed(2) +
                ', center=(' + c.center_x.toFixed(2) + ', ' + c.center_y.toFixed(2) + ')' +
                '</div>';
    }});
    pitsEl.innerHTML = html;
}}

document.getElementById('info').innerHTML =
    document.getElementById('info').innerHTML.replace('{{pointsCount}}', pointsData.length.toString());

// === Анимация ===
function animate() {{
    requestAnimationFrame(animate);
    points.rotation.z += 0.001;
    planeMeshBottom.rotation.copy(points.rotation);
    if (planeMeshTop) planeMeshTop.rotation.copy(points.rotation);
    perpendicularsGroup.rotation.copy(points.rotation);
    renderer.render(scene, camera);
}}
animate();
</script>

</body>
</html>
"""

    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML визуализация '{html_filename}' успешно создана!")

def find_clusters_on_projection(X, Y, Z, threshold=3.0, min_cluster_size=10):
    """
    Находит кластеры на проекции (сетке Z), где Z > threshold.
    Возвращает:
        labeled_Z  - матрица меток кластеров той же формы, что Z (0 - нет кластера)
        num_labels - количество кластеров
        clusters_info - список словарей с краткой информацией о кластерах
    """

    # Маска превышения порога
    mask = (Z > threshold)
    
    # Обработка NaN (интерполяция могла дать NaN за пределами)
    mask = np.logical_and(mask, ~np.isnan(Z))
    print("\nОтладка маски для кластеров:")
    print("  Всего пикселей в Z:", Z.size)
    print("  NaN в Z:", np.isnan(Z).sum())
    print("  Пикселей Z > threshold:", (Z > threshold).sum())
    print("  Маска после Z > threshold и !NaN:", mask.sum())
    # Поиск связных компонент (4-связность по умолчанию)
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=bool)
    labeled, num_labels = label(mask, structure=structure)

    clusters_info = []
    for lbl in range(1, num_labels + 1):
        indices = np.where(labeled == lbl)
        size = len(indices[0])
        # if size < min_cluster_size:
        #     # Кластеры меньше порога размера считаем шумом, сбрасываем их в 0
        #     labeled[indices] = 0
        #     continue

        z_vals = Z[indices]
        x_vals = X[indices]
        y_vals = Y[indices]

        clusters_info.append({
            "label": lbl,
            "size": int(size),
            "mean_height": float(np.nanmean(z_vals)),
            "max_height": float(np.nanmax(z_vals)),
            "center_x": float(np.nanmean(x_vals)),
            "center_y": float(np.nanmean(y_vals)),
        })

    # Переиндексация меток (после удаления мелких кластеров)
    unique_labels = np.unique(labeled)
    unique_labels = unique_labels[unique_labels != 0]
    remap = {old: new for new, old in enumerate(unique_labels)}
    remap[0] = 0
    labeled_remap = np.zeros_like(labeled, dtype=int)
    for old, new in remap.items():
        labeled_remap[labeled == old] = new

    # Обновляем labels в clusters_info
    new_clusters_info = []
    for c in clusters_info:
        old_lbl = c["label"]
        if old_lbl in remap:
            c["label"] = remap[old_lbl]
            new_clusters_info.append(c)

    print(f"\nНайдено кластеров (после фильтрации по размеру): {len(new_clusters_info)}")
    for c in new_clusters_info:
        print(f"  Кластер {c['label']}: size={c['size']}, "
              f"max_height={c['max_height']:.3f}, "
              f"center=({c['center_x']:.2f}, {c['center_y']:.2f})")
    
    return labeled_remap, len(new_clusters_info), new_clusters_info

def assign_cluster_ids_to_points(points, X, Y, labeled_Z):
    """
    Назначает каждой исходной точке id кластера по её (x, y),
    используя ближайший узел сетки X, Y и матрицу меток labeled_Z.
    
    Возвращает:
        cluster_ids: массив int, длина = len(points), -1 если точка вне кластеров.
    """
    x_grid = X[0, :]
    y_grid = Y[:, 0]

    # Быстрая функция поиска ближайшего индекса в отсортированном массиве
    def find_nearest_idx(arr, val):
        idx = np.searchsorted(arr, val)
        if idx <= 0:
            return 0
        if idx >= len(arr):
            return len(arr) - 1
        # сравниваем два ближайших
        if abs(arr[idx] - val) < abs(arr[idx-1] - val):
            return idx
        else:
            return idx - 1

    cluster_ids = np.full(points.shape[0], -1, dtype=int)

    for i, (x, y, _) in enumerate(points):
        ix = find_nearest_idx(x_grid, x)
        iy = find_nearest_idx(y_grid, y)
        label_val = labeled_Z[iy, ix]  # внимание на порядок (y,x)
        if label_val > 0:
            cluster_ids[i] = label_val
        else:
            cluster_ids[i] = -1

    print(f"\nТочек, попавших в кластеры: {(cluster_ids >= 0).sum()} "
          f"из {len(cluster_ids)}")
    return cluster_ids

def debug_some_cluster_points(points, X, Y, labeled_Z, cluster_ids, n_samples=20):
    print("\nПримеры точек, попавших в кластеры:")
    idxs = np.where(cluster_ids >= 0)[0]
    if len(idxs) == 0:
        print("  Нет точек с cluster_id >= 0")
        return
    idxs = idxs[:min(len(idxs), n_samples)]

    x_grid = X[0, :]
    y_grid = Y[:, 0]

    def find_nearest_idx(arr, val):
        idx = np.searchsorted(arr, val)
        if idx <= 0:
            return 0
        if idx >= len(arr):
            return len(arr) - 1
        return idx if abs(arr[idx] - val) < abs(arr[idx-1] - val) else idx-1

    for i in idxs:
        x, y, z = points[i]
        ix = find_nearest_idx(x_grid, x)
        iy = find_nearest_idx(y_grid, y)
        label_val = labeled_Z[iy, ix]
        print(f"  point #{i}: (x={x:.2f}, y={y:.2f}, z={z:.2f}), "
              f"cluster_id={cluster_ids[i]}, labeled_Z={label_val}")

def debug_point_region(X, Y, Z, threshold, x0, y0, radius=0.2):
    """
    Печатает статистику по Z и маске в окрестности точки (x0, y0).
    """
    dist = np.sqrt((X - x0)**2 + (Y - y0)**2)
    region = dist < radius

    if not np.any(region):
        print(f"В радиусе {radius} вокруг ({x0}, {y0}) нет узлов сетки.")
        return

    Z_region = Z[region]
    print(f"\nОтладка региона вокруг ({x0:.2f}, {y0:.2f}), radius={radius}:")
    print("  Кол-во узлов:", region.sum())
    print("  NaN в регионе:", np.isnan(Z_region).sum())
    print("  Z_min:", np.nanmin(Z_region))
    print("  Z_max:", np.nanmax(Z_region))
    print("  Пикселей Z > threshold:", np.logical_and(~np.isnan(Z_region), Z_region > threshold).sum())

def fit_upper_plane_to_points(points):
    """
    Находит плоскость z = m x + n y + p,
    которая НЕ пересекает облако точек (лежит сверху) и
    минимизирует максимальное отклонение (Chebyshev fit).

    Условия:
        (m xi + n yi + p) - zi >= 0  для всех i  (точки не выше плоскости)
        (m xi + n yi + p) - zi <= t  для всех i  (ограничиваем расстояние снизу)
        t >= 0
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    n_points = len(points)

    A = np.c_[X, Y, np.ones(n_points)]
    m0, n0, p0 = np.linalg.lstsq(A, Z, rcond=None)[0]
    x0 = np.array([m0, n0, p0, 1.0])

    def obj(x):
        return x[3]

    def obj_grad(x):
        return np.array([0.0, 0.0, 0.0, 1.0])

    cons = []

    # (m xi + n yi + p) - zi >= 0
    def make_g_i(i):
        xi, yi, zi = X[i], Y[i], Z[i]
        def g_i(x):
            m, n, p, t = x
            return (m*xi + n*yi + p) - zi
        def g_i_jac(x):
            return np.array([xi, yi, 1.0, 0.0])
        return {'type': 'ineq', 'fun': g_i, 'jac': g_i_jac}

    # t - [(m xi + n yi + p) - zi] >= 0
    def make_h_i(i):
        xi, yi, zi = X[i], Y[i], Z[i]
        def h_i(x):
            m, n, p, t = x
            return t - ((m*xi + n*yi + p) - zi)
        def h_i_jac(x):
            # d/d[m,n,p,t] (t - m xi - n yi - p + zi) = [-xi,-yi,-1,1]
            return np.array([-xi, -yi, -1.0, 1.0])
        return {'type': 'ineq', 'fun': h_i, 'jac': h_i_jac}

    for i in range(n_points):
        cons.append(make_g_i(i))
        cons.append(make_h_i(i))

    # t >= 0
    def g_t(x):
        return x[3]
    def g_t_jac(x):
        return np.array([0.0, 0.0, 0.0, 1.0])
    cons.append({'type': 'ineq', 'fun': g_t, 'jac': g_t_jac})

    from scipy.optimize import minimize
    res = minimize(
        obj, x0,
        method='SLSQP',
        jac=obj_grad,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
    )

    if not res.success:
        print("ВНИМАНИЕ: оптимизация (верхняя плоскость) не сошлась, используем МНК-плоскость.")
        m, n, p = m0, n0, p0
    else:
        m, n, p, t = res.x
        print(f"\nОптимизация (верхняя плоскость) завершена. t (макс. отклонение): {t:.4f}")

    print(f"\nПараметры верхней плоскости (без пересечения облака):")
    print(f"z = {m:.4f}*x + {n:.4f}*y + {p:.4f}")
    
    return m, n, p

def calculate_perpendicular_distances_to_upper_plane(points, plane_params):
    """
    Для верхней плоскости z = m x + n y + p,
    расстояние вниз: d = ((m x0 + n y0 + p) - z0) / sqrt(1 + m^2 + n^2) >= 0.
    """
    m, n, p = plane_params
    z_plane = m * points[:, 0] + n * points[:, 1] + p
    distances_raw = (z_plane - points[:, 2]) / np.sqrt(1 + m**2 + n**2)
    distances = np.maximum(distances_raw, 0.0)

    print(f"\nСтатистика отклонений вниз (верхняя плоскость, >= 0):")
    print(f"Минимальное: {np.min(distances):.4f}")
    print(f"Максимальное: {np.max(distances):.4f}")
    print(f"Среднее: {np.mean(distances):.4f}")
    print(f"Стандартное отклонение: {np.std(distances):.4f}")
    
    return distances
def visualize_threshold_clusters_2d(X, Y, Z, labeled_Z, threshold=3.0,
                                   filename_prefix="clusters_binary",
                                   clusters_info=None):
    """
    Делает 2D-визуализацию с жёстким порогом:
      - Z <= threshold -> зелёный
      - Z >  threshold -> красный
    и поверх рисует контуры кластеров (labeled_Z).

    Параметры:
        X, Y, Z       - сетка (как из create_relief_map)
        labeled_Z     - метки кластеров (0 - фон, 1..N - кластеры)
        threshold     - порог по Z
        filename_prefix - префикс имени файла
        clusters_info - список словарей с инфо о кластерах (опционально),
                        каждый словарь должен иметь 'label', 'center_x', 'center_y'
    """

    from matplotlib.colors import ListedColormap

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Бинарная карта: 0 - ниже/равно порогу, 1 - выше порога
    binary = np.zeros_like(Z, dtype=int)
    mask_valid = ~np.isnan(Z)
    binary[np.logical_and(mask_valid, Z > threshold)] = 1

    print("\nОтладка бинарной карты для 2D-визуализации:")
    print("  Всего узлов:", Z.size)
    print("  NaN в Z:", np.isnan(Z).sum())
    print("  Пикселей Z > threshold:", (Z > threshold).sum())
    print("  Пикселей в binary == 1:", (binary == 1).sum())

    # Двухцветная карта: 0 -> зелёный, 1 -> красный
    cmap_bin = ListedColormap(["#00aa00", "#ff0000"])

    plt.figure(figsize=(12, 10))
    im = plt.imshow(
        binary,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='equal',
        cmap=cmap_bin,
        vmin=0, vmax=1
    )
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Z ≤ порог", "Z > порог"])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Двухпороговая карта (порог={threshold}) и кластеры")

    # Контуры кластеров по labeled_Z (0 - фон, 1..N - кластеры)
    mask_clusters = (labeled_Z > 0)
    plt.contour(
        X, Y, mask_clusters.astype(int),
        levels=[0.5],
        colors='black',
        linewidths=1.0
    )

    # Подписи центров кластеров, если передан clusters_info
    if clusters_info is not None:
        for c in clusters_info:
            plt.text(c["center_x"], c["center_y"],
                     f"{c['label']}",
                     color="black",
                     fontsize=8,
                     ha="center", va="center",
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    out = f"{filename_prefix}_binary_with_clusters.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Двухпороговая карта с кластерами сохранена в '{out}'")

if __name__ == "__main__":
    print("=" * 60)
    print("АНАЛИЗ РЕЛЬЕФА 3D МОДЕЛИ")
    print("=" * 60)
    
    # 1. Генерируем облако точек
    print("\n1. Генерация облака точек...")
    ply_file, points = generate_rounded_cube_point_cloud(
        filename='cube_half.ply',
        size=10.0,
        bend_depth=3.0,
        points_per_edge=80,
        hole_radius=2.0
    )
    
    # 2. Нижняя опорная плоскость (для пиков)
    print("\n2. Поиск нижней опорной плоскости (под облаком)...")
    plane_params_bottom = fit_plane_to_points(points)
    
    # 3. Расстояния от нижней плоскости (как было – «пики»)
    print("\n3. Вычисление перпендикулярных расстояний (пики)...")
    distances_bottom = calculate_perpendicular_distances(points, plane_params_bottom)
    
    # 4. Карта рельефа для пиков
    print("\n4. Создание карты рельефа (пики)...")
    X_peaks, Y_peaks, Z_peaks = create_relief_map(points, distances_bottom)

    # 5. Кластеры пиков на проекции (Z_peaks > threshold)
    print("\n5. Поиск кластеров пиков на проекции (Z_peaks > 4.0)...")
    labeled_Z_peaks, n_clusters_peaks, clusters_info_peaks = find_clusters_on_projection(
        X_peaks, Y_peaks, Z_peaks, threshold=4.0, min_cluster_size=5
    )

    # 5a. 2D-визуализация кластеров пиков
    print("\n5a. Визуализация кластеров пиков на 2D-проекции...")
    visualize_clusters_2d(X_peaks, Y_peaks, Z_peaks, labeled_Z_peaks,
                          filename_prefix="clusters_projection_peaks")

    # 5b. Двухпороговая визуализация для пиков
    print("\n5b. Двухпороговая визуализация (пики, зелёный/красный)...")
    visualize_threshold_clusters_2d(
        X_peaks, Y_peaks, Z_peaks, labeled_Z_peaks,
        threshold=4.0,
        filename_prefix="clusters_projection_peaks"
    )

    # 6. Назначаем кластеры пиков исходным 3D точкам
    print("\n6. Назначение кластеров пиков исходным точкам...")
    cluster_ids_peaks = assign_cluster_ids_to_points(points, X_peaks, Y_peaks, labeled_Z_peaks)
    debug_some_cluster_points(points, X_peaks, Y_peaks, labeled_Z_peaks, cluster_ids_peaks)

    # -----------------------------------------------------------
    # ВЕРХНЯЯ ПЛОСКОСТЬ И ВПАДИНЫ
    # -----------------------------------------------------------
    print("\n2b. Поиск верхней опорной плоскости (над облаком)...")
    plane_params_top = fit_upper_plane_to_points(points)

    print("\n3b. Вычисление перпендикулярных расстояний вниз до верхней плоскости (впадины)...")
    distances_top = calculate_perpendicular_distances_to_upper_plane(points, plane_params_top)

    print("\n4b. Создание карты рельефа для впадин...")
    X_pits, Y_pits, Z_pits = create_relief_map(points, distances_top,
                                               resolution=200, hole_radius=2.0)

    print("\n5b. Поиск кластеров впадин на проекции (Z_pits > 4.0)...")
    labeled_Z_pits, n_clusters_pits, clusters_info_pits = find_clusters_on_projection(
        X_pits, Y_pits, Z_pits, threshold=4.0, min_cluster_size=5
    )

    print("\n5c. Визуализация кластеров впадин на 2D-проекции...")
    visualize_clusters_2d(X_pits, Y_pits, Z_pits, labeled_Z_pits,
                          filename_prefix="clusters_projection_pits")

    print("\n5d. Двухпороговая визуализация (впадины, зелёный/красный)...")
    visualize_threshold_clusters_2d(
        X_pits, Y_pits, Z_pits, labeled_Z_pits,
        threshold=4.0,
        filename_prefix="clusters_projection_pits"
    )

    print("\n6b. Назначение кластеров впадин исходным точкам...")
    cluster_ids_pits = assign_cluster_ids_to_points(points, X_pits, Y_pits, labeled_Z_pits)
    debug_some_cluster_points(points, X_pits, Y_pits, labeled_Z_pits, cluster_ids_pits)

    # -----------------------------------------------------------
    # 3D визуализация: пики и впадины разными цветами
    # -----------------------------------------------------------
    print("\n7. Создание 3D визуализации с пиками и впадинами...")
    create_html_visualization(
        ply_file,
        points,
        plane_params_bottom,        # нижняя плоскость
        distances_bottom,           # расстояния для статистики пиков
        plane_params_top=plane_params_top,  # верхняя плоскость
        distances_top=distances_top,        # расстояния для статистики впадин
        cluster_ids_peaks=cluster_ids_peaks,
        clusters_info_peaks=clusters_info_peaks,
        cluster_ids_pits=cluster_ids_pits,
        clusters_info_pits=clusters_info_pits,
        html_filename='visualization_with_plane_peaks_pits.html'
    )
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print("\nФайлы созданы:")
    print("  - cube_half.ply - облако точек")
    print("  - relief_map_peaks.png / clusters_projection_peaks_*.png - пики")
    print("  - relief_map_pits.png  / clusters_projection_pits_*.png  - впадины")
    print("  - visualization_with_plane_peaks_pits.html - интерактивная 3D визуализация")
    print("\nОткройте 'visualization_with_plane_peaks_pits.html' в браузере для просмотра.")
    print("Пики подсвечиваются жёлтым, впадины — голубым.")
