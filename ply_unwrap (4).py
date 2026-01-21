import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def geodesic_unwrap_debug(ply_file, output_image='unwrapped.png', resolution=2000, 
                          num_cuts=2, z_scale_factor=1.0, debug=True):
    """
    Геодезическая развёртка с сохранением рельефа
    
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
    z_scale_factor : float
        Коэффициент масштабирования вертикали (больше 1 = растягивает, меньше 1 = сжимает)
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
    vertex_data = plydata['vertex'].data
    has_color = 'red' in vertex_data.dtype.names
    if has_color:
        intensity = np.column_stack([vertices['red'], 
                                    vertices['green'], 
                                    vertices['blue']]) / 255.0
        print("Найдены RGB данные")
    else:
        r_temp = np.sqrt(x**2 + y**2)
        intensity = r_temp
        print("RGB отсутствует, используем радиус для визуализации")
    
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
    print(f"Отношение осей: {eigenvalues[0]/eigenvalues[1]:.2f}:{eigenvalues[1]/eigenvalues[2]:.2f}")
    
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
    print(f"r (радиус): min={r.min():.3f}, max={r.max():.3f}, mean={r.mean():.3f}, std={r.std():.3f}")
    print(f"theta (угол): min={np.degrees(theta.min()):.1f}°, max={np.degrees(theta.max()):.1f}°")
    print(f"Вариация радиуса: {(r.max()-r.min())/r.mean()*100:.1f}%")
    
    if debug:
        # Визуализация 1: Исходная модель с разными проекциями
        fig = plt.figure(figsize=(18, 10))
        
        # 3D вид с углом
        ax1 = fig.add_subplot(231, projection='3d')
        scatter = ax1.scatter(x_rot, y_rot, z_rot, c=theta, cmap='hsv', s=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D модель (цвет = угол θ)')
        plt.colorbar(scatter, ax=ax1, label='Угол θ (рад)')
        
        # 3D вид с радиусом
        ax2 = fig.add_subplot(232, projection='3d')
        scatter2 = ax2.scatter(x_rot, y_rot, z_rot, c=r, cmap='viridis', s=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('3D модель (цвет = радиус r)')
        plt.colorbar(scatter2, ax=ax2, label='Радиус r')
        
        # Проекция θ-Z с радиусом
        ax3 = fig.add_subplot(233)
        scatter3 = ax3.scatter(np.degrees(theta), z_rot, c=r, s=1, cmap='viridis')
        ax3.set_xlabel('Угол θ (градусы)')
        ax3.set_ylabel('Высота Z')
        ax3.set_title('Проекция θ-Z (цвет = радиус)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Радиус r')
        
        # Проекция r-Z с углом
        ax4 = fig.add_subplot(234)
        scatter4 = ax4.scatter(r, z_rot, c=theta, s=1, cmap='hsv')
        ax4.set_xlabel('Радиус r')
        ax4.set_ylabel('Высота Z')
        ax4.set_title('Проекция r-Z (цвет = угол θ)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Угол θ (рад)')
        
        # Плотность в θ-Z
        ax5 = fig.add_subplot(235)
        hist, xedges, yedges = np.histogram2d(theta, z_rot, bins=100)
        ax5.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                  extent=[np.degrees(theta.min()), np.degrees(theta.max()), 
                         z_rot.min(), z_rot.max()])
        ax5.set_xlabel('Угол θ (градусы)')
        ax5.set_ylabel('Высота Z')
        ax5.set_title('Плотность точек в θ-Z')
        
        # Радиус вдоль высоты
        ax6 = fig.add_subplot(236)
        z_bins = np.linspace(z_rot.min(), z_rot.max(), 50)
        r_stats = []
        for i in range(len(z_bins)-1):
            mask = (z_rot >= z_bins[i]) & (z_rot < z_bins[i+1])
            if np.any(mask):
                r_stats.append([z_bins[i], r[mask].mean(), r[mask].min(), r[mask].max()])
        r_stats = np.array(r_stats)
        ax6.plot(r_stats[:, 1], r_stats[:, 0], 'b-', label='Средний r', linewidth=2)
        ax6.fill_betweenx(r_stats[:, 0], r_stats[:, 2], r_stats[:, 3], 
                         alpha=0.3, label='min-max')
        ax6.set_xlabel('Радиус r')
        ax6.set_ylabel('Высота Z')
        ax6.set_title('Профиль радиуса по высоте')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig('debug_01_original.png', dpi=150)
        print("\n[DEBUG] Сохранено: debug_01_original.png")
    
    print("\n" + "=" * 60)
    print(f"ЭТАП 3: Разрезание и развёртка (ваш метод)")
    print("=" * 60)
    
    # Углы разрезов
    cut_angles = np.linspace(-np.pi, np.pi, num_cuts + 1)
    print(f"Углы разрезов: {np.degrees(cut_angles)}")
    
    all_u = []
    all_v = []
    all_intensity = []
    all_section_id = []
    all_original_r = []
    all_original_z = []
    all_original_theta = []
    
    for i in range(num_cuts):
        angle_start = cut_angles[i]
        angle_end = cut_angles[i + 1]
        
        # Маска для точек в диапазоне
        if i == num_cuts - 1:
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
        
        r_mean = np.mean(r_section)
        r_min = np.min(r_section)
        r_max = np.max(r_section)
        print(f"  Радиус: mean={r_mean:.3f}, min={r_min:.3f}, max={r_max:.3f}, var={(r_max-r_min)/r_mean*100:.1f}%")
        
        # ВАШ МЕТОД: u = r * theta, v = z * (r / r_mean)
        u = r_section * theta_section
        v = z_section * (r_section / r_mean) * z_scale_factor
        
        print(f"  UV после развёртки:")
        print(f"    u: min={u.min():.3f}, max={u.max():.3f}, range={u.max()-u.min():.3f}")
        print(f"    v: min={v.min():.3f}, max={v.max():.3f}, range={v.max()-v.min():.3f}")
        
        all_u.extend(u)
        all_v.extend(v)
        all_original_r.extend(r_section)
        all_original_z.extend(z_section)
        all_original_theta.extend(theta_section)
        
        if len(intensity.shape) == 1:
            all_intensity.extend(intensity_section)
        else:
            all_intensity.extend(intensity_section.tolist())
        
        all_section_id.extend([i] * n_points)
    
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_intensity = np.array(all_intensity)
    all_section_id = np.array(all_section_id)
    all_original_r = np.array(all_original_r)
    all_original_z = np.array(all_original_z)
    all_original_theta = np.array(all_original_theta)
    
    print(f"\nВсего точек после разрезания: {len(all_u)}")
    print(f"u: min={all_u.min():.3f}, max={all_u.max():.3f}, range={all_u.max()-all_u.min():.3f}")
    print(f"v: min={all_v.min():.3f}, max={all_v.max():.3f}, range={all_v.max()-all_v.min():.3f}")
    
    if debug:
        # Визуализация 2: UV координаты с детальным анализом
        fig = plt.figure(figsize=(18, 12))
        
        # UV по секциям
        ax1 = fig.add_subplot(231)
        scatter1 = ax1.scatter(all_u, all_v, c=all_section_id, s=1, cmap='tab10')
        ax1.set_xlabel('u')
        ax1.set_ylabel('v')
        ax1.set_title('UV координаты (цвет = секция)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter1, ax=ax1, label='Секция')
        
        # UV с радиусом
        ax2 = fig.add_subplot(232)
        scatter2 = ax2.scatter(all_u, all_v, c=all_original_r, s=1, cmap='viridis')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        ax2.set_title('UV координаты (цвет = исходный радиус)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter2, ax=ax2, label='Радиус r')
        
        # UV с углом
        ax3 = fig.add_subplot(233)
        scatter3 = ax3.scatter(all_u, all_v, c=all_original_theta, s=1, cmap='hsv')
        ax3.set_xlabel('u')
        ax3.set_ylabel('v')
        ax3.set_title('UV координаты (цвет = угол θ)')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter3, ax=ax3, label='Угол θ')
        
        # Плотность в UV
        ax4 = fig.add_subplot(234)
        hist, xedges, yedges = np.histogram2d(all_u, all_v, bins=200)
        im = ax4.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                       extent=[all_u.min(), all_u.max(), all_v.min(), all_v.max()])
        ax4.set_xlabel('u')
        ax4.set_ylabel('v')
        ax4.set_title('Плотность точек в UV')
        plt.colorbar(im, ax=ax4, label='Количество точек')
        
        # Сравнение трансформации Z -> v
        ax5 = fig.add_subplot(235)
        scatter5 = ax5.scatter(all_original_z, all_v, c=all_original_r, s=1, 
                             cmap='viridis', alpha=0.5)
        ax5.plot([all_original_z.min(), all_original_z.max()], 
                [all_original_z.min(), all_original_z.max()], 
                'r--', linewidth=2, label='v=z (без масштабирования)')
        ax5.set_xlabel('Исходная Z')
        ax5.set_ylabel('Новая v')
        ax5.set_title('Трансформация Z → v (цвет = радиус)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter5, ax=ax5, label='Радиус r')
        
        # Профиль v вдоль u для каждой секции
        ax6 = fig.add_subplot(236)
        for sec_id in range(num_cuts):
            mask = all_section_id == sec_id
            if np.any(mask):
                ax6.scatter(all_u[mask], all_v[mask], s=1, label=f'Секция {sec_id+1}')
        ax6.set_xlabel('u')
        ax6.set_ylabel('v')
        ax6.set_title('Профили секций')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
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
    print(f"Соотношение сторон UV: {(u_max-u_min)/(v_max-v_min):.3f}")
    
    # Создание сетки
    grid_u, grid_v = np.mgrid[0:1:complex(0, resolution), 
                               0:1:complex(0, resolution)]
    
    points = np.column_stack([u_norm, v_norm])
    
    print(f"\nИнтерполяция на сетку {resolution}x{resolution}...")
    
    # Интерполяция с ближайшим соседом для отладки
    if debug:
        grid_nearest_section = griddata(points, all_section_id, 
                                       (grid_u, grid_v), 
                                       method='nearest')
        grid_nearest_r = griddata(points, all_original_r,
                                 (grid_u, grid_v),
                                 method='nearest')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        im1 = axes[0].imshow(grid_nearest_section.T, origin='lower', cmap='tab10', 
                           aspect='auto', interpolation='nearest')
        axes[0].set_title('Ближайший сосед: секции')
        axes[0].set_xlabel('u (нормализованный)')
        axes[0].set_ylabel('v (нормализованный)')
        plt.colorbar(im1, ax=axes[0], label='Номер секции')
        
        im2 = axes[1].imshow(grid_nearest_r.T, origin='lower', cmap='viridis',
                           aspect='auto', interpolation='nearest')
        axes[1].set_title('Ближайший сосед: радиус (рельеф)')
        axes[1].set_xlabel('u (нормализованный)')
        axes[1].set_ylabel('v (нормализованный)')
        plt.colorbar(im2, ax=axes[1], label='Радиус r')
        
        plt.tight_layout()
        plt.savefig('debug_03_nearest_neighbor.png', dpi=150)
        print("\n[DEBUG] Сохранено: debug_03_nearest_neighbor.png")
    
    # Основная интерполяция
    if len(all_intensity.shape) == 1:
        print("Интерполяция одноканального изображения (радиус)...")
        grid_intensity = griddata(points, all_intensity, 
                                 (grid_u, grid_v), 
                                 method='linear')
        
        # Заполняем NaN
        nan_mask = np.isnan(grid_intensity)
        if np.any(nan_mask):
            print(f"ВНИМАНИЕ: {np.sum(nan_mask)} пикселей ({np.sum(nan_mask)/(resolution**2)*100:.1f}%) содержат NaN")
            print("  Заполняем методом nearest neighbor...")
            grid_intensity_nearest = griddata(points, all_intensity,
                                             (grid_u, grid_v),
                                             method='nearest')
            grid_intensity[nan_mask] = grid_intensity_nearest[nan_mask]
        
        plt.figure(figsize=(18, 10))
        plt.imshow(grid_intensity.T, origin='lower', cmap='terrain', 
                  aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Рельеф (радиус)', fraction=0.046, pad=0.04)
        
    else:
        print("Интерполяция RGB изображения...")
        grid_rgb = np.zeros((resolution, resolution, 3))
        for channel in range(3):
            grid_channel = griddata(points, 
                                   all_intensity[:, channel], 
                                   (grid_u, grid_v), 
                                   method='linear')
            
            nan_mask = np.isnan(grid_channel)
            if np.any(nan_mask):
                grid_nearest = griddata(points, all_intensity[:, channel],
                                       (grid_u, grid_v), method='nearest')
                grid_channel[nan_mask] = grid_nearest[nan_mask]
            
            grid_rgb[:, :, channel] = grid_channel
        
        plt.figure(figsize=(18, 10))
        plt.imshow(grid_rgb.T, origin='lower', aspect='auto', 
                  interpolation='bilinear')
    
    plt.xlabel('u (развёрнутая окружность)', fontsize=12)
    plt.ylabel('v (высота с учётом радиуса)', fontsize=12)
    plt.title(f'Геодезическая развёртка ({num_cuts} секций, z_scale={z_scale_factor})', 
             fontsize=14)
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"\n[РЕЗУЛЬТАТ] Развёртка сохранена в {output_image}")
    
    print("\n" + "=" * 60)
    print("ОТЛАДКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"\nИспользованный метод: u = r * theta, v = z * (r/r_mean) * {z_scale_factor}")
    print("Параметр z_scale_factor можно регулировать для управления вертикальным масштабом")
    
    return grid_intensity if len(all_intensity.shape) == 1 else grid_rgb


# Использование
if __name__ == "__main__":
    ply_file = "your_model.ply"
    
    # Ваш метод с возможностью регулировки вертикального масштаба
    unwrapped = geodesic_unwrap_debug(
        ply_file, 
        output_image='unwrapped_user_method.png',
        resolution=2000,
        num_cuts=2,
        z_scale_factor=1.0,  # Регулируйте этот параметр для изменения пропорций
        debug=True
    )
    
    # Попробуйте разные масштабы если нужно:
    # z_scale_factor=0.5  - сжать вертикаль вдвое
    # z_scale_factor=2.0  - растянуть вертикаль вдвое
