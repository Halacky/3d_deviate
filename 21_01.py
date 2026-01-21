import numpy as np
import json
import matplotlib.pyplot as plt

from scipy import ndimage  # для label
import os


def generate_cylinder_with_defects(
    R=1.0,
    H=5.0,
    n_theta=400,
    n_z=400,
    bend_amp=0.2,
    bend_width=0.5,
    defects_params=None,
    seed=42
):
    """
    Генерация облака точек цилиндра без доньев, с загнутыми краями и 3 дефектами.
    """
    np.random.seed(seed)

    # Параметрическая сетка
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, H, n_z)

    theta_grid, z_grid = np.meshgrid(thetas, zs, indexing="ij")  # (n_theta, n_z)

    # Загнутые края (снизу и сверху)
    bend_bottom = bend_amp * np.exp(-(z_grid / bend_width) ** 2)
    bend_top = bend_amp * np.exp(-((H - z_grid) / bend_width) ** 2)
    R_bent = R + bend_bottom + bend_top  # (n_theta, n_z)

    # Если не заданы дефекты — задаём 3 «нароста»
    if defects_params is None:
        defects_params = [
            {
                "theta_center": 0.3 * 2 * np.pi,
                "z_center": 0.3 * H,
                "amp": 0.3,
                "sigma_theta": 0.15,
                "sigma_z": 0.4,
            },
            {
                "theta_center": 0.75 * 2 * np.pi,
                "z_center": 0.6 * H,
                "amp": 0.4,
                "sigma_theta": 0.18,
                "sigma_z": 0.3,
            },
            {
                "theta_center": 1.5 * np.pi,
                "z_center": 0.15 * H,
                "amp": 0.25,
                "sigma_theta": 0.12,
                "sigma_z": 0.35,
            },
        ]

    delta_R_def = np.zeros_like(R_bent)

    # Добавляем дефекты (наросты) на поверхности
    for d in defects_params:
        theta_c = d["theta_center"]
        z_c = d["z_center"]
        amp = d["amp"]
        sigma_theta = d["sigma_theta"]
        sigma_z = d["sigma_z"]

        # Разность углов с учётом периодичности ([-pi, pi])
        dtheta = np.angle(np.exp(1j * (theta_grid - theta_c)))
        dz = z_grid - z_c

        gauss = np.exp(-((dtheta / sigma_theta) ** 2 + (dz / sigma_z) ** 2))
        delta_R_def += amp * gauss

    # Итоговый радиус
    R_final = R_bent + delta_R_def

    # 3D координаты
    X = R_final * np.cos(theta_grid)
    Y = R_final * np.sin(theta_grid)
    Z = z_grid

    points3d = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # 2D развертка (параметрическое пространство)
    # u = theta / (2*pi) in [0,1), v = z/H in [0,1]
    theta_norm = theta_grid / (2.0 * np.pi)
    z_norm = z_grid / H
    U = theta_norm
    V = z_norm

    points2d = np.stack([U.ravel(), V.ravel()], axis=1)

    # Карта высоты дефектов (без загиба краёв)
    defect_height = delta_R_def

    return points3d, points2d, defects_params, U, V, defect_height


def write_ply_points(filename, points, comments=None):
    """
    Запись точек (x, y, z) или (x, y) в PLY как облако точек.
    Если points имеет форму (N, 2), z = 0.
    """
    points = np.asarray(points)
    n_points = points.shape[0]

    if points.shape[1] == 2:
        pts = np.zeros((n_points, 3), dtype=np.float32)
        pts[:, :2] = points
    elif points.shape[1] == 3:
        pts = points.astype(np.float32)
    else:
        raise ValueError("points must be of shape (N, 2) or (N, 3)")

    header_lines = [
        "ply",
        "format ascii 1.0",
    ]

    if comments:
        for c in comments:
            header_lines.append(f"comment {c}")

    header_lines.extend([
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ])

    with open(filename, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_ply_points_with_colors(filename, points, colors, comments=None):
    """
    Запись точек (x, y, z) с цветом (r,g,b) в PLY.
    points.shape = (N, 3)
    colors.shape = (N, 3), значения 0..255
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    n_points = points.shape[0]
    assert points.shape == (n_points, 3)
    assert colors.shape == (n_points, 3)

    header_lines = [
        "ply",
        "format ascii 1.0",
    ]

    if comments:
        for c in comments:
            header_lines.append(f"comment {c}")

    header_lines.extend([
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    with open(filename, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def save_unwrap_image(filename, U, V, defect_height):
    """
    Создание изображения 2D-развертки.
    """
    h = defect_height
    h_min, h_max = np.min(h), np.max(h)
    if h_max > h_min:
        h_norm = (h - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(h)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.imshow(
        h_norm.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Relative defect height")
    plt.xlabel("u = theta / (2π)")
    plt.ylabel("v = z / H")
    plt.title("2D Unwrap of Cylinder Surface (Defects Map)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------
#  Поиск дефектов на 2D-карте
# ---------------------------

def detect_defects_on_unwrap(defect_height,
                             threshold_rel=0.5,
                             min_size_pixels=20,
                             connectivity=1):
    """
    Поиск дефектов на 2D-карте defect_height.
    Возвращает:
        labels_2d: 2D массив int тех же размеров, 0 — фон, 1..N — кластеры дефектов
        num_labels: количество кластеров
    Параметры:
        threshold_rel: относительный порог по высоте.
            0.5 означает, что все точки с высотой > h_min + 0.5*(h_max - h_min)
            считаются кандидатами на дефект.
        min_size_pixels: минимальный размер кластера в пикселях (точках сетки),
            меньше — отбрасываем как шум.
        connectivity: связность для label (1 или 2).
    """
    h = defect_height
    h_min, h_max = np.min(h), np.max(h)
    if h_max == h_min:
        # нет вариаций — дефектов нет
        labels = np.zeros_like(h, dtype=np.int32)
        return labels, 0

    # Бинарная маска "кандидатов в дефекты"
    thr = h_min + threshold_rel * (h_max - h_min)
    mask = h > thr

    # Маркируем связные компоненты
    labeled, num = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, connectivity))

    if num == 0:
        return labeled, 0

    # Удаляем слишком маленькие компоненты
    labels = labeled.copy()
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))  # количество пикселей в каждом кластере

    # создаём таблицу remap: если размер < min_size_pixels -> 0 (фон)
    label_map = {0: 0}
    new_label_id = 1
    for idx, size in enumerate(sizes, start=1):
        if size >= min_size_pixels:
            label_map[idx] = new_label_id
            new_label_id += 1
        else:
            label_map[idx] = 0

    # пересчитываем метки
    vectorized_map = np.vectorize(lambda x: label_map.get(x, 0), otypes=[np.int32])
    labels = vectorized_map(labels)
    final_num_labels = new_label_id - 1

    return labels, final_num_labels


def visualize_defects_on_unwrap(filename, defect_height, labels_2d):
    """
    Сохраняем картинку с картой дефектов и раскрашенными кластерами.
    """
    h = defect_height
    h_min, h_max = np.min(h), np.max(h)
    if h_max > h_min:
        h_norm = (h - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(h)

    plt.figure(figsize=(10, 4), dpi=150)

    # 1) слева - исходная карта высоты
    plt.subplot(1, 2, 1)
    plt.imshow(h_norm.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    plt.title("Defect height (normalized)")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.colorbar()

    # 2) справа - маска кластеров (каждый кластер - свой цвет)
    plt.subplot(1, 2, 2)
    # labels_2d: (n_theta, n_z). Транспонируем для отображения по тем же осям.
    plt.imshow(labels_2d.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="tab20")
    plt.title("Detected defect clusters")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def map_defect_labels_to_3d(labels_2d, n_theta, n_z):
    """
    Преобразуем 2D метки (n_theta, n_z) в вектор меток для всех 3D-точек,
    учитывая, что points3d сформированы как ravel() по (theta, z):

        idx = i_theta * n_z + i_z

    Возвращает:
        labels_1d: вектор длины n_theta*n_z с метками дефектов (0..N).
    """
    assert labels_2d.shape == (n_theta, n_z)
    # Уплощаем в том же порядке, как была создана points3d: X.ravel(), Y.ravel(), Z.ravel()
    labels_1d = labels_2d.ravel(order='C')  # по умолчанию C-order: сначала ось 0 (theta), потом ось 1 (z)
    return labels_1d


def create_colored_pointcloud_from_labels(points3d, labels_1d):
    """
    Назначаем цвет каждой 3D-точке в зависимости от метки дефекта.
    - фон (label=0) -> серый цвет
    - дефекты (label>0) -> разные яркие цвета
    """
    n_points = points3d.shape[0]
    colors = np.zeros((n_points, 3), dtype=np.uint8)

    # фон: серый
    colors[:, :] = np.array([180, 180, 180], dtype=np.uint8)

    # Подберём 20 разных цветов (циклически)
    palette = np.array([
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
        [255, 128,   0],
        [128,   0, 255],
        [  0, 128, 255],
        [128, 255,   0],
        [255,   0, 128],
        [  0, 255, 128],
        [128, 128, 255],
        [128, 255, 128],
        [255, 128, 128],
        [255, 255, 128],
        [255, 128, 255],
        [128, 255, 255],
        [ 64,  64, 255],
        [255,  64,  64],
    ], dtype=np.uint8)

    labels_unique = np.unique(labels_1d)
    labels_unique = labels_unique[labels_unique > 0]  # только дефекты

    for i, label in enumerate(labels_unique):
        color = palette[i % len(palette)]
        colors[labels_1d == label] = color

    return colors

from mpl_toolkits.mplot3d import Axes3D  # нужно для 3D-проекций

def show_pointcloud_matplotlib(points3d, colors=None, stride=4):
    """
    Простая визуализация 3D-облака точек через matplotlib.
    stride - шаг по точкам (чтобы не рисовать все, если их много).
    """
    pts = points3d[::stride]
    if colors is not None:
        cols = colors[::stride] / 255.0  # matplotlib ждёт [0..1]
    else:
        cols = 'b'

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1, 1, 1])  # одинаковый масштаб по осям

    plt.tight_layout()
    plt.show()


import open3d as o3d
import numpy as np

def show_pointcloud_open3d(points3d, colors=None):
    """
    Визуализация point cloud через open3d.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)

    if colors is not None:
        # цвета в диапазоне [0,1]
        cols = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)

    o3d.visualization.draw_geometries([pcd])

import plotly.offline as pyo
import plotly.graph_objects as go


def save_pointcloud_html(filename, points3d, colors=None, title="3D point cloud"):
    """
    Сохранение 3D-облака точек в HTML (Plotly, оффлайн).
    points3d: (N,3)
    colors: (N,3) uint8 или None.
        Если colors=None, точки будут одинакового цвета.
    """
    pts = np.asarray(points3d)
    assert pts.shape[1] == 3

    if colors is not None:
        cols = np.asarray(colors, dtype=np.uint8)
        assert cols.shape == pts.shape
        # Plotly принимает цвета в формате 'rgb(r,g,b)'
        color_strings = [
            f"rgb({r},{g},{b})" for r, g, b in cols
        ]
    else:
        color_strings = 'rgb(100,100,100)'

    trace = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_strings,
            opacity=0.8,
        )
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # одинаковый масштаб по осям
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Сохраняем как один самодостаточный HTML-файл
    pyo.plot(
        fig,
        filename=filename,
        auto_open=False,    # не пытаться открыть браузер на сервере
        include_plotlyjs='cdn'  # или 'cdn' / 'directory'; см. комментарий ниже
    )
if __name__ == "__main__":
    # Создадим директорию вывода (по желанию)
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. Генерация модели
    n_theta = 400
    n_z = 400
    points3d, points2d, defects, U, V, defect_height = generate_cylinder_with_defects(
        R=1.0,
        H=5.0,
        n_theta=n_theta,
        n_z=n_z,
        bend_amp=0.2,
        bend_width=0.5,
    )

    # 2. Сохранение 3D PLY (без цветов)
    write_ply_points(
        os.path.join(out_dir, "cylinder_with_defects_3d.ply"),
        points3d,
        comments=["Cylindrical shell with bent edges and 3 defects"],
    )

    # 3. Сохранение 2D развертки в PLY (u,v -> x,y, z=0)
    write_ply_points(
        os.path.join(out_dir, "cylinder_with_defects_2d_unwrap.ply"),
        points2d,
        comments=[
            "2D unwrap of cylinder",
            "u = theta / (2*pi) in [0,1), v = z/H in [0,1]",
        ],
    )

    # 4. Сохранение параметров дефектов
    with open(os.path.join(out_dir, "defects_params.json"), "w", encoding="utf-8") as f:
        json.dump(defects, f, ensure_ascii=False, indent=2)

    # 5. Создание изображения развертки (с картой дефектов)
    save_unwrap_image(os.path.join(out_dir, "cylinder_unwrap_defects.png"), U, V, defect_height)

    # 6. Поиск дефектов на 2D-карте
    labels_2d, num_labels = detect_defects_on_unwrap(
        defect_height,
        threshold_rel=0.5,   # относительный порог
        min_size_pixels=50,  # отсечение мелких пятен
        connectivity=1
    )

    print(f"Найдено кластеров дефектов: {num_labels}")

    # 7. Визуализация дефектов на 2D-развертке
    visualize_defects_on_unwrap(
        os.path.join(out_dir, "cylinder_unwrap_defects_clusters.png"),
        defect_height,
        labels_2d
    )

    labels_1d = map_defect_labels_to_3d(labels_2d, n_theta=n_theta, n_z=n_z)

    # 9. Создаём цветное облако точек
    colors = create_colored_pointcloud_from_labels(points3d, labels_1d)

    write_ply_points_with_colors(
        os.path.join(out_dir, "cylinder_with_defects_3d_colored.ply"),
        points3d,
        colors,
        comments=[
            "Cylindrical shell with detected defect clusters",
            "Defects are color-highlighted"
        ],
    )

    # 10. HTML-визуализация (оффлайн)
    html_path = os.path.join(out_dir, "cylinder_with_defects_3d_colored.html")
    save_pointcloud_html(
        html_path,
        points3d,
        colors,
        title="Cylinder with detected defect clusters"
    )

    print("Готово:")
    print(" - cylinder_with_defects_3d.ply")
    print(" - cylinder_with_defects_2d_unwrap.ply")
    print(" - cylinder_unwrap_defects.png")
    print(" - cylinder_unwrap_defects_clusters.png")
    print(" - cylinder_with_defects_3d_colored.ply")
    print(" - cylinder_with_defects_3d_colored.html")
    print(" - defects_params.json")
