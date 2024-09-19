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

def wind_simulation(grid: np.ndarray, wind_data):
    def map_colors(data, cmap="viridis"):
        data = np.array(data)
        
        # Initialize an array for RGB colors with the same shape as the input data
        rgb_array = np.zeros((data.shape[0], data.shape[1], 3))  # RGB has 3 channels
        
        # Step 1: Filter out zero values
        non_zero_mask = data > 0
        non_zero_values = data[non_zero_mask]
        
        # Check if there are any non-zero values
        if non_zero_values.size == 0:
            return rgb_array  # Return an array with all zeros if no non-zero values
        
        # Step 2: Normalize the non-zero values
        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)
        norm = cplt.Normalize(vmin=min_val, vmax=max_val)
        
        # Step 3: Create a color map
        cmap = plt.get_cmap(cmap)
        
        # Map non-zero values to colors
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if non_zero_mask[i, j]:
                    rgb_array[i, j] = cmap(norm(data[i, j]))[:3]  # Extract RGB and ignore alpha
        
        return rgb_array

    def map_colors_result(coast, grid):
        result = np.zeros((grid.shape[0], grid.shape[1], 3))
        colors = map_colors(coast)

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if colors[x][y][0] != 0 and colors[x][y][1] != 0 and colors[x][y][1] != 0: result[x][y] = 255 * colors[x][y]
                elif grid[x][y] == 1: result[x][y] = [30, 30, 30]
                else: result[x][y] = [0, 0 ,0]

        return result

    def last_index_of_item(lst, item):
        try:
            return len(lst) - 1 - lst[::-1].index(item)
        except ValueError:
            return None

    height, width = grid.shape
    result = np.zeros_like(grid)
    angles = range(180, 360)

    i = 0
    for angle in angles:
        i += 1
        print(f"{round(100 * i / len(angles), 2):6.2f}%", end="\r")

        wind_constant = wind_data.loc[angle, "freq"] * wind_data.loc[angle, "speed"]

        dx = np.cos(np.radians(angle))
        dy = np.sin(np.radians(angle))
        for pos in [[i, height - 1] for i in range(0, width - 1)] + [[0, i], for i in range()]:
            hit = False
            while (0 <= pos[0] < width and 0 <= pos[1] < height) and not hit:
                if grid[math.floor(pos[1]), math.floor(pos[0])] == 1:
                    result[math.floor(pos[1]), math.floor(pos[0])] += wind_constant
                    hit = True

                pos[0] += dx
                pos[1] += dy

    result = map_colors_result(result, grid)
    return result

def main():
    coastline = process_image("map.png", {"red > 200": 1, "red < 200": 0})

    df = pd.read_csv("skarpöHData.csv", delimiter=";", usecols=["Datum", "Vindriktning", "Vindhastighet"])
    df = df.dropna()
    df["Datum"] = pd.to_datetime(df["Datum"])

    wind_data = df.groupby("Vindriktning").agg(
        freq = ("Vindriktning", "count"),
        speed = ("Vindhastighet", "mean")
    )

    exposure_map = wind_simulation(coastline, wind_data)
    save_exposure_map(exposure_map)

if __name__ == "__main__":
    main()
