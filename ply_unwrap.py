import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def unwrap_cylindrical_mesh(ply_file, output_image='unwrapped.png', resolution=1000):
    """
    Развёртка цилиндрической PLY модели в 2D изображение
    
    Parameters:
    -----------
    ply_file : str
        Путь к PLY файлу
    output_image : str
        Путь для сохранения результата
    resolution : int
        Разрешение выходного изображения
    """
    
    # Загрузка PLY файла
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']
    
    # Извлечение координат
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    
    # Определение оси цилиндра (обычно Z, но можем определить автоматически)
    # Находим главную ось через PCA
    coords = np.column_stack([x, y, z])
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    
    # Вычисляем ковариационную матрицу
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Главная ось - собственный вектор с наибольшим собственным значением
    main_axis_idx = np.argmax(eigenvalues)
    main_axis = eigenvectors[:, main_axis_idx]
    
    # Поворачиваем модель так, чтобы главная ось совпадала с Z
    if main_axis_idx != 2:
        # Создаём матрицу поворота
        z_axis = np.array([0, 0, 1])
        v = np.cross(main_axis, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(main_axis, z_axis)
        
        if s > 1e-6:  # Если оси не параллельны
            vx = np.array([[0, -v[2], v[1]], 
                          [v[2], 0, -v[0]], 
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
            coords_centered = coords_centered @ R.T
    
    x_rot = coords_centered[:, 0]
    y_rot = coords_centered[:, 1]
    z_rot = coords_centered[:, 2]
    
    # Преобразование в цилиндрические координаты
    r = np.sqrt(x_rot**2 + y_rot**2)
    theta = np.arctan2(y_rot, x_rot)  # угол от -π до π
    height = z_rot
    
    # Нормализация для развёртки
    # theta -> координата X (разворачиваем по окружности)
    # height -> координата Y
    theta_normalized = (theta + np.pi) / (2 * np.pi)  # 0 до 1
    
    height_min, height_max = height.min(), height.max()
    height_normalized = (height - height_min) / (height_max - height_min)
    
    # Используем радиус как значение интенсивности (рельеф)
    # Можно также использовать другие атрибуты если они есть
    if 'red' in vertices.dtype.names:
        # Если есть цвет, используем его
        colors = np.column_stack([vertices['red'], 
                                 vertices['green'], 
                                 vertices['blue']]) / 255.0
        intensity = colors
    else:
        # Иначе используем радиус как рельеф
        intensity = r
    
    # Создание регулярной сетки для изображения
    grid_x, grid_y = np.mgrid[0:1:complex(0, resolution), 
                               0:1:complex(0, resolution)]
    
    # Интерполяция значений на регулярную сетку
    points = np.column_stack([theta_normalized, height_normalized])
    
    if len(intensity.shape) == 1:
        # Одноканальное изображение (рельеф)
        grid_z = griddata(points, intensity, (grid_x, grid_y), 
                         method='linear', fill_value=0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_z.T, origin='lower', cmap='viridis', 
                  aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Радиус (рельеф)')
        plt.xlabel('Угол θ (развёрнутый)')
        plt.ylabel('Высота Z')
        plt.title('Развёртка цилиндрической модели')
    else:
        # RGB изображение
        grid_rgb = np.zeros((resolution, resolution, 3))
        for i in range(3):
            grid_rgb[:, :, i] = griddata(points, intensity[:, i], 
                                        (grid_x, grid_y), 
                                        method='linear', fill_value=0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_rgb.T, origin='lower', aspect='auto', 
                  interpolation='bilinear')
        plt.xlabel('Угол θ (развёрнутый)')
        plt.ylabel('Высота Z')
        plt.title('Развёртка цилиндрической модели')
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Развёртка сохранена в {output_image}")
    
    return grid_z if len(intensity.shape) == 1 else grid_rgb


# Использование
if __name__ == "__main__":
    # Укажите путь к вашему PLY файлу
    ply_file = "your_model.ply"
    
    unwrapped = unwrap_cylindrical_mesh(
        ply_file, 
        output_image='unwrapped_cylinder.png',
        resolution=2000  # Увеличьте для лучшего качества
    )
