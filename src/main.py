import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cplt
from PIL import Image
import math


def save_exposure_map(exposure_map: np.ndarray, output_path='exposure.png'):
    image = Image.fromarray(exposure_map.astype(np.uint8))
    image.save(output_path)


def process_image(image_path, color_conditions):
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size

    result = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            pixel_value = 0

            for condition, action in color_conditions.items():
                channel, operator, value = condition.split()
                value = int(value)
                channel_value = {"red": r, "green": g, "blue": b}.get(channel.lower())

                if operator == ">" and channel_value > value:
                    pixel_value = max(pixel_value, action)
                elif operator == "<" and channel_value < value:
                    pixel_value = max(pixel_value, action)
                elif operator == "==" and channel_value == value:
                    pixel_value = max(pixel_value, action)

            result[y, x] = pixel_value

    return result

def wind_simulation(grid: np.ndarray, wind_data, angles=range(180, 360), thickness=1):
    def map_colors(data, cmap="viridis"):
        data = np.array(data)
        rgb_array = np.zeros((data.shape[0], data.shape[1], 3))
        
        non_zero_mask = data > 0
        non_zero_values = data[non_zero_mask]
        
        if non_zero_values.size == 0:
            return rgb_array
        
        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)
        norm = cplt.Normalize(vmin=min_val, vmax=max_val)
        
        cmap = plt.get_cmap(cmap)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if non_zero_mask[i, j]:
                    rgb_array[i, j] = cmap(norm(data[i, j]))[:3]
        
        return rgb_array

    def map_colors_result(coast, grid, thickness=1):
        result = np.zeros((grid.shape[0], grid.shape[1], 3))
        colors = map_colors(coast, "viridis")

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if colors[x][y][0] != 0 and colors[x][y][1] != 0 and colors[x][y][1] != 0:
                    for i in range(thickness):
                        if y - i > 0:
                            result[x][y - i] = 255 * colors[x][y]

                elif grid[x][y] == 1: result[x][y] = [30, 30, 30]
                else: result[x][y] = [0, 0, 0]

        return result

    def last_index(lst, item) -> int:
        return len(lst) - 1 - lst[::-1].index(item)

    height, width = grid.shape
    result = np.zeros_like(grid)

    i = 0
    for angle in angles:
        i += 1
        print(f"{round(100 * i / len(angles), 2):6.2f}%", end="\r")

        wind_constant = wind_data.loc[angle, "freq"] * wind_data.loc[angle, "speed"]

        dx = np.cos(np.radians(angle))
        dy = np.sin(np.radians(angle))

        sparsity_y = int(max(1, abs(np.tan(np.radians(angle)))))
        sparsity_x = int(max(1, abs(np.tan(np.radians(angle - 90)))))
        if sparsity_x == 0:
            bottom_pos = []
        else:
            bottom_pos = [[j, height - 1] for j in range(0, width - 1, sparsity_x)]

        if sparsity_y == 0:
            left_pos, right_pos = [], [] 
        else:
            left_pos = [[0, j] for j in range(last_index(list(grid[:, 0]), 1) + 10, height - 1, sparsity_y)]
            right_pos = [[width - 1, j] for j in range(last_index(list(grid[:, width - 1]), 1), height - 1, sparsity_y)]

        for pos in bottom_pos + left_pos + right_pos:
            hit = False
            while (0 <= pos[0] < width and 0 <= pos[1] < height) and not hit:
                if grid[math.floor(pos[1]), math.floor(pos[0])] == 1:
                    hit = True

                    result[math.floor(pos[1]), math.floor(pos[0])] += wind_constant
                    wind_overwrite = result[math.floor(pos[1]), math.floor(pos[0])]

                    # Apply thickness
                    for _ in range(thickness - 1):
                        if 0 <= pos[0] < width and 0 <= pos[1] < height:
                            result[math.floor(pos[1]), math.floor(pos[0])] = wind_overwrite
                            pos[0] -= dx; pos[1] += dy

                pos[0] -= dx
                pos[1] += dy

    for i in range(15):
        result = np.delete(result, i, axis=1)
        result = np.delete(result, result.shape[1] - 1 - i, axis=1)
        grid = np.delete(grid, i, axis=1)
        grid = np.delete(grid, grid.shape[1] - 1 - i, axis=1)

    result = map_colors_result(result, grid)
    return result

def main():
    coastline = process_image("input/map_highres.png", {"red > 200": 1, "red < 200": 0})
    coastline = np.delete(coastline, 0, axis=1)
    coastline = np.delete(coastline, coastline.shape[1] - 1, axis=1)

    df = pd.read_csv("input/wind_proc.csv")
    df = df.dropna()
    df["dt"] = pd.to_datetime(df["dt"])

    # Data from the last 10 years
    wind_data = df[df["dt"] > pd.to_datetime("2014-01-01")].groupby("winddirection").agg(
        freq = ("winddirection", "count"),
        speed = ("windspeed", "mean")
    )

    exposure_map = wind_simulation(coastline, wind_data, thickness=3)
    save_exposure_map(exposure_map, "output/exposure.png")

if __name__ == "__main__":
    main()
