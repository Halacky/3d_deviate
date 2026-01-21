import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def geodesic_unwrap_debug(ply_file, output_image='unwrapped.png', resolution=2000, num_cuts=2, debug=True):
    """
    Геодезическая развёртка с детальной отладкой
    
    Parameters:
    -----------
    ply_file : str
        Путь к PLY файлу
    output_image : str
        Путь для сохранения результата
    resolution : int
        Разрешение выходного изображения
    num_cuts : int
        Количество вертикальных разрезов
    debug : bool
        Включить отладочные графики
    """
    
    print("=" * 60)
    print("ЭТАП 1: Загрузка данных")
    print("=" * 60)
    
    # Загрузка PLY
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']
    
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])
    
    print(f"Загружено вершин: {len(x)}")
    print(f"X: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")
    print(f"Y: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    print(f"Z: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}")
    
    coords = np.column_stack([x, y, z])
    
    # Получаем цвет/интенсивность
    has_color = 'red' in vertices.dtype.names
    if has_color:
        intensity = np.column_stack([vertices['red'], 
                                    vertices['green'], 
                                    vertices['blue']]) / 255.0
        print("Найдены RGB данные")
    else:
        r = np.sqrt(x**2 + y**2)
        intensity = r
        print("RGB отсутствует, используем радиус")
    
    print("\n" + "=" * 60)
    print("ЭТАП 2: Центрирование и выравнивание")
    print("=" * 60)
    
    # Центрируем модель
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    print(f"Центроид: {centroid}")
    
    # PCA для определения главной оси
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx].real
    eigenvectors = eigenvectors[:, sorted_idx].real
    
    print(f"Собственные значения: {eigenvalues}")
    print(f"Главная ось: {eigenvectors[:, 0]}")
    
    # Выравниваем по Z
    main_axis = eigenvectors[:, 0]
    z_axis = np.array([0, 0, 1])
    
    if not np.allclose(main_axis, z_axis):
        v = np.cross(main_axis, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(main_axis, z_axis)
        
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], 
                          [v[2], 0, -v[0]], 
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
            coords_centered = coords_centered @ R.T
            print("Применён поворот для выравнивания по Z")
    
    x_rot = coords_centered[:, 0]
    y_rot = coords_centered[:, 1]
    z_rot = coords_centered[:, 2]
    
    print(f"После выравнивания:")
    print(f"X: min={x_rot.min():.3f}, max={x_rot.max():.3f}")
    print(f"Y: min={y_rot.min():.3f}, max={y_rot.max():.3f}")
    print(f"Z: min={z_rot.min():.3f}, max={z_rot.max():.3f}")
    
    # Вычисляем цилиндрические координаты
    r = np.sqrt(x_rot**2 + y_rot**2)
    theta = np.arctan2(y_rot, x_rot)
    
    print(f"\nЦилиндрические координаты:")
    print(f"r (радиус): min={r.min():.3f}, max={r.max():.3f}, mean={r.mean():.3f}")
    print(f"theta (угол): min={np.degrees(theta.min()):.1f}°, max={np.degrees(theta.max()):.1f}°")
    
    if debug:
        # Визуализация 1: Исходная модель
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(x_rot, y_rot, z_rot, c=theta, cmap='hsv', s=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Модель (цвет = угол θ)')
        plt.colorbar(scatter, ax=ax1, label='Угол θ (рад)')
        
        ax2 = fig.add_subplot(132)
        ax2.scatter(theta, z_rot, c=r, s=1, cmap='viridis')
        ax2.set_xlabel('Угол θ (рад)')
        ax2.set_ylabel('Высота Z')
        ax2.set_title('Проекция θ-Z (цвет = радиус)')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(133)
        hist, xedges, yedges = np.histogram2d(theta, z_rot, bins=100)
        ax3.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                  extent=[theta.min(), theta.max(), z_rot.min(), z_rot.max()])
        ax3.set_xlabel('Угол θ (рад)')
        ax3.set_ylabel('Высота Z')
        ax3.set_title('Плотность точек')
        
        plt.tight_layout()
        plt.savefig('debug_01_original.png', dpi=150)
        print("\n[DEBUG] Сохранено: debug_01_original.png")
    
    print("\n" + "=" * 60)
    print("ЭТАП 3: Разрезание на секции")
    print("=" * 60)
    
    # Углы разрезов
    cut_angles = np.linspace(-np.pi, np.pi, num_cuts + 1)
    print(f"Углы разрезов: {np.degrees(cut_angles)}")
    
    all_u = []
    all_v = []
    all_intensity = []
    all_section_id = []
    
    for i in range(num_cuts):
        angle_start = cut_angles[i]
        angle_end = cut_angles[i + 1]
        
        # Маска для точек в диапазоне
        if i == num_cuts - 1:
            # Последняя секция включает конец
            mask = (theta >= angle_start) & (theta <= angle_end)
        else:
            mask = (theta >= angle_start) & (theta < angle_end)
        
        n_points = np.sum(mask)
        print(f"\nСекция {i+1}: угол от {np.degrees(angle_start):.1f}° до {np.degrees(angle_end):.1f}°")
        print(f"  Точек в секции: {n_points}")
        
        if n_points == 0:
            print(f"  ВНИМАНИЕ: Секция пуста!")
            continue
        
        # Точки текущей секции
        theta_section = theta[mask]
        z_section = z_rot[mask]
        r_section = r[mask]
        
        if len(intensity.shape) == 1:
            intensity_section = intensity[mask]
        else:
            intensity_section = intensity[mask, :]
        
        # Простая развёртка: theta -> u, z -> v
        # Масштабируем theta в дуговую длину
        r_mean = np.mean(r_section)
        print(f"  Средний радиус: {r_mean:.3f}")
        
        # u - это дуговая длина вдоль окружности на среднем радиусе
        u = r_mean * theta_section
        v = z_section
        
        print(f"  u: min={u.min():.3f}, max={u.max():.3f}, range={u.max()-u.min():.3f}")
        print(f"  v: min={v.min():.3f}, max={v.max():.3f}, range={v.max()-v.min():.3f}")
        
        all_u.extend(u)
        all_v.extend(v)
        
        if len(intensity.shape) == 1:
            all_intensity.extend(intensity_section)
        else:
            all_intensity.extend(intensity_section.tolist())
        
        all_section_id.extend([i] * n_points)
    
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_intensity = np.array(all_intensity)
    all_section_id = np.array(all_section_id)
    
    print(f"\nВсего точек после разрезания: {len(all_u)}")
    print(f"u: min={all_u.min():.3f}, max={all_u.max():.3f}")
    print(f"v: min={all_v.min():.3f}, max={all_v.max():.3f}")
    
    if debug:
        # Визуализация 2: UV координаты по секциям
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # График UV координат с раскраской по секциям
        scatter1 = axes[0].scatter(all_u, all_v, c=all_section_id, s=1, cmap='tab10')
        axes[0].set_xlabel('u (дуговая длина)')
        axes[0].set_ylabel('v (высота)')
        axes[0].set_title('UV координаты (цвет = номер секции)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        plt.colorbar(scatter1, ax=axes[0], label='Секция')
        
        # Гистограмма плотности
        hist, xedges, yedges = np.histogram2d(all_u, all_v, bins=200)
        im = axes[1].imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                           extent=[all_u.min(), all_u.max(), all_v.min(), all_v.max()])
        axes[1].set_xlabel('u (дуговая длина)')
        axes[1].set_ylabel('v (высота)')
        axes[1].set_title('Плотность точек в UV')
        plt.colorbar(im, ax=axes[1], label='Количество точек')
        
        plt.tight_layout()
        plt.savefig('debug_02_uv_coordinates.png', dpi=150)
        print("\n[DEBUG] Сохранено: debug_02_uv_coordinates.png")
    
    print("\n" + "=" * 60)
    print("ЭТАП 4: Нормализация и интерполяция")
    print("=" * 60)
    
    # Нормализация
    u_min, u_max = all_u.min(), all_u.max()
    v_min, v_max = all_v.min(), all_v.max()
    
    u_norm = (all_u - u_min) / (u_max - u_min)
    v_norm = (all_v - v_min) / (v_max - v_min)
    
    print(f"Нормализованные координаты:")
    print(f"u_norm: min={u_norm.min():.3f}, max={u_norm.max():.3f}")
    print(f"v_norm: min={v_norm.min():.3f}, max={v_norm.max():.3f}")
    
    # Создание сетки
    grid_u, grid_v = np.mgrid[0:1:complex(0, resolution), 
                               0:1:complex(0, resolution)]
    
    points = np.column_stack([u_norm, v_norm])
    
    print(f"\nИнтерполяция на сетку {resolution}x{resolution}...")
    
    # Интерполяция с ближайшим соседом для отладки
    if debug:
        grid_nearest = griddata(points, all_section_id, 
                               (grid_u, grid_v), 
                               method='nearest')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        im = ax.imshow(grid_nearest.T, origin='lower', cmap='tab10', 
                      aspect='auto', interpolation='nearest')
        ax.set_title('Интерполяция: метод ближайшего соседа (секции)')
        ax.set_xlabel('u (нормализованный)')
        ax.set_ylabel('v (нормализованный)')
        plt.colorbar(im, ax=ax, label='Номер секции')
        plt.tight_layout()
        plt.savefig('debug_03_nearest_neighbor.png', dpi=150)
        print("\n[DEBUG] Сохранено: debug_03_nearest_neighbor.png")
    
    # Основная интерполяция
    if len(all_intensity.shape) == 1:
        print("Интерполяция одноканального изображения...")
        grid_intensity = griddata(points, all_intensity, 
                                 (grid_u, grid_v), 
                                 method='linear')
        
        # Заполняем NaN средним значением
        nan_mask = np.isnan(grid_intensity)
        if np.any(nan_mask):
            print(f"ВНИМАНИЕ: {np.sum(nan_mask)} пикселей содержат NaN")
            grid_intensity[nan_mask] = np.nanmean(grid_intensity)
        
        plt.figure(figsize=(16, 10))
        plt.imshow(grid_intensity.T, origin='lower', cmap='terrain', 
                  aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Рельеф')
        
    else:
        print("Интерполяция RGB изображения...")
        grid_rgb = np.zeros((resolution, resolution, 3))
        for channel in range(3):
            grid_rgb[:, :, channel] = griddata(points, 
                                              all_intensity[:, channel], 
                                              (grid_u, grid_v), 
                                              method='linear',
                                              fill_value=0)
        
        # Заполняем NaN
        for channel in range(3):
            nan_mask = np.isnan(grid_rgb[:, :, channel])
            if np.any(nan_mask):
                grid_rgb[nan_mask, channel] = 0
        
        plt.figure(figsize=(16, 10))
        plt.imshow(grid_rgb.T, origin='lower', aspect='auto', 
                  interpolation='bilinear')
    
    plt.xlabel('u (развёрнутая окружность)')
    plt.ylabel('v (высота)')
    plt.title(f'Геодезическая развёртка ({num_cuts} секций)')
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"\n[РЕЗУЛЬТАТ] Развёртка сохранена в {output_image}")
    
    print("\n" + "=" * 60)
    print("ОТЛАДКА ЗАВЕРШЕНА")
    print("=" * 60)
    
    return grid_intensity if len(all_intensity.shape) == 1 else grid_rgb


# Использование
if __name__ == "__main__":
    ply_file = "your_model.ply"
    
    unwrapped = geodesic_unwrap_debug(
        ply_file, 
        output_image='unwrapped_final.png',
        resolution=2000,
        num_cuts=2,
        debug=True  # Включить отладку
    )
