import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

def geodesic_unwrap(ply_file, output_image='unwrapped.png', resolution=2000, num_cuts=2):
    """
    Геодезическая развёртка 3D модели через разрезание и проекцию
    
    Parameters:
    -----------
    ply_file : str
        Путь к PLY файлу
    output_image : str
        Путь для сохранения результата
    resolution : int
        Разрешение выходного изображения
    num_cuts : int
        Количество вертикальных разрезов (обычно 2 для двух половинок)
    """
    
    # Загрузка PLY
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']
    
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])
    
    coords = np.column_stack([x, y, z])
    
    # Центрируем модель
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    
    # Определяем главную ось (обычно Z для вытянутых объектов)
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_axis_idx = np.argmax(eigenvalues)
    
    # Выравниваем по Z если нужно
    if main_axis_idx != 2:
        main_axis = eigenvectors[:, main_axis_idx].real
        z_axis = np.array([0, 0, 1])
        v = np.cross(main_axis, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(main_axis, z_axis)
        
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], 
                          [v[2], 0, -v[0]], 
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
            coords_centered = coords_centered @ R.T
    
    x_rot = coords_centered[:, 0]
    y_rot = coords_centered[:, 1]
    z_rot = coords_centered[:, 2]
    
    # Получаем цвет/интенсивность для визуализации рельефа
    if 'red' in vertices.dtype.names:
        intensity = np.column_stack([vertices['red'], 
                                    vertices['green'], 
                                    vertices['blue']]) / 255.0
    else:
        # Используем радиус как индикатор рельефа
        r = np.sqrt(x_rot**2 + y_rot**2)
        intensity = r
    
    # Разрезаем модель на секции
    theta = np.arctan2(y_rot, x_rot)
    
    # Создаём углы разрезов
    cut_angles = np.linspace(-np.pi, np.pi, num_cuts + 1)
    
    all_u = []
    all_v = []
    all_intensity = []
    
    for i in range(num_cuts):
        # Выбираем точки для текущей секции
        angle_start = cut_angles[i]
        angle_end = cut_angles[i + 1]
        
        # Маска для точек в этом диапазоне углов
        mask = (theta >= angle_start) & (theta < angle_end)
        
        if not np.any(mask):
            continue
        
        # Точки текущей секции
        x_section = x_rot[mask]
        y_section = y_rot[mask]
        z_section = z_rot[mask]
        theta_section = theta[mask]
        
        if len(intensity.shape) == 1:
            intensity_section = intensity[mask]
        else:
            intensity_section = intensity[mask, :]
        
        # Геодезическая проекция (похожа на равнопромежуточную)
        # Локальная система координат для секции
        mid_angle = (angle_start + angle_end) / 2
        
        # Вращаем секцию так, чтобы середина смотрела вперёд
        cos_a = np.cos(-mid_angle)
        sin_a = np.sin(-mid_angle)
        
        x_local = x_section * cos_a - y_section * sin_a
        y_local = x_section * sin_a + y_section * cos_a
        
        # Радиус в локальных координатах
        r_local = np.sqrt(x_local**2 + y_local**2)
        
        # Геодезическая развёртка: 
        # u - это дуговая длина вдоль окружности
        # v - это высота Z
        theta_local = np.arctan2(y_local, x_local)
        
        # Вычисляем дуговую длину (как на сфере)
        # Используем средний радиус секции
        r_mean = np.mean(r_local)
        
        u = r_mean * theta_local  # Дуговая длина
        v = z_section
        
        # Смещаем каждую секцию горизонтально
        u_offset = i * 2 * np.pi * r_mean / num_cuts
        
        all_u.extend(u + u_offset)
        all_v.extend(v)
        
        if len(intensity.shape) == 1:
            all_intensity.extend(intensity_section)
        else:
            all_intensity.extend(intensity_section.tolist())
    
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_intensity = np.array(all_intensity)
    
    # Нормализация координат
    u_min, u_max = all_u.min(), all_u.max()
    v_min, v_max = all_v.min(), all_v.max()
    
    u_norm = (all_u - u_min) / (u_max - u_min)
    v_norm = (all_v - v_min) / (v_max - v_min)
    
    # Создание регулярной сетки
    grid_u, grid_v = np.mgrid[0:1:complex(0, resolution), 
                               0:1:complex(0, resolution)]
    
    # Интерполяция
    points = np.column_stack([u_norm, v_norm])
    
    if len(all_intensity.shape) == 1:
        # Одноканальное
        grid_intensity = griddata(points, all_intensity, 
                                 (grid_u, grid_v), 
                                 method='linear', fill_value=np.nan)
        
        plt.figure(figsize=(16, 10))
        plt.imshow(grid_intensity.T, origin='lower', cmap='terrain', 
                  aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Рельеф (радиус)')
        
    else:
        # RGB
        grid_rgb = np.zeros((resolution, resolution, 3))
        for channel in range(3):
            grid_rgb[:, :, channel] = griddata(points, 
                                              all_intensity[:, channel], 
                                              (grid_u, grid_v), 
                                              method='linear', 
                                              fill_value=0)
        
        plt.figure(figsize=(16, 10))
        plt.imshow(grid_rgb.T, origin='lower', aspect='auto', 
                  interpolation='bilinear')
    
    # Добавляем линии разрезов
    for i in range(1, num_cuts):
        cut_position = i / num_cuts
        plt.axvline(x=cut_position * resolution, color='red', 
                   linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Развёрнутая окружность (дуговая длина)')
    plt.ylabel('Высота Z')
    plt.title(f'Геодезическая развёртка ({num_cuts} секций)')
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Развёртка сохранена в {output_image}")
    
    return grid_intensity if len(all_intensity.shape) == 1 else grid_rgb


# Использование
if __name__ == "__main__":
    # Пример использования
    ply_file = "your_model.ply"
    
    # С двумя половинками (как вы предложили)
    unwrapped = geodesic_unwrap(
        ply_file, 
        output_image='unwrapped_geodesic.png',
        resolution=2000,
        num_cuts=2  # Две половинки
    )
    
    # Или с большим количеством секций для более точной развёртки
    # unwrapped = geodesic_unwrap(ply_file, num_cuts=4, resolution=2000)
