import numpy as np
import pygame
import pandas as pd
from PIL import Image

# Visualize as blue to red instead of black to white
def save_exposure_map(exposure_map, output_path='exposure_map.png'):
    exposure_min = np.min(exposure_map)
    exposure_max = np.max(exposure_map)

    if exposure_max > exposure_min:
        normalized_map = (exposure_map - exposure_min) / (exposure_max - exposure_min) * 255
    else:
        normalized_map = np.zeros_like(exposure_map)

    exposure_image = Image.fromarray(normalized_map.astype(np.uint8), mode='L')
    exposure_image.save(output_path)


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

def simulate_wind(grid, wind_angles):
    height, width = grid.shape
    result = np.zeros_like(grid)
    wind_angles_rad = np.deg2rad(wind_angles)
    
    def find_closest_coastline(x, y, dx, dy):
        for dist in range(1, height):
            nx = x + int(dx * dist)
            ny = y + int(dy * dist)
            
            if 0 <= nx < width and 0 <= ny < height:
                return nx, ny
            elif 0 > nx >= width or 0 > ny >= height:
                return nx, ny
            elif grid[ny, nx] == 1:
                return nx, ny
            elif grid[ny, nx] == 0:
                continue
        return None, None
    
    for angle in wind_angles_rad:
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        
        for x in range(width):
            for y in range(height):
                # Skip ocean points
                if grid[y, x] == 0:
                    continue
                
                dx = cos_angle
                dy = sin_angle
                
                closest_x, closest_y = find_closest_coastline(x, y, dx, dy)
                
                if closest_x is not None and closest_y is not None:
                    result[closest_y, closest_x] += 1
    
    return result

def pg(grid: np.ndarray):
    def colorize(g) -> np.ndarray:
        h, w = g.shape
        clr_grid = np.zeros((h, w, 3), dtype=np.uint8)
        clr_grid[g >= 1] = [255, 255, 255]
        clr_grid[g == 0] = [0, 0, 0]
        return clr_grid


    def draw_color_grid(surface, color_grid):
        for x in range(len(color_grid)):
            for y in range(len(color_grid[0])):
                color = color_grid[x][y]
                surface.set_at((y, x), color)

    pygame.init()
    height, width = grid.shape
    window = pygame.display.set_mode((width, height))

    result = np.zeros_like(grid)
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        window.fill((0, 0, 0))

        result = np.zeros_like(grid)
        angle = 0
        rays_pos = [[i, height] for i in range(0, width)]
        for pos in rays_pos:
            c = 0;
            while c < height:
                c += 1
                pos[1] -= 1
                if grid[pos[1], pos[0]] == 1:
                    result[pos[1], pos[0]] = 1; c = 1000

        draw_color_grid(window, colorize(result))

        pygame.display.update()

def main():
    coastline = process_image("map.png", {"red > 200": 1, "red < 200": 0})

    df = pd.read_csv("skarpöHData.csv", delimiter=";", usecols=["Datum", "Vindriktning", "Vindhastighet"])
    df = df.dropna()
    df["Datum"] = pd.to_datetime(df["Datum"])

    exposure_map = pg(coastline)

    save_exposure_map(exposure_map)

if __name__ == "__main__":
    main()
