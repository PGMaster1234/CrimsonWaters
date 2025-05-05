styles = ["regular", "bold", "thin", "extralight"]
sizes = [i for i in range(1, 100)] + [200]

# Dictionary to hold font definitions: {name: (path, size)}
fonts = {}

font_dir = "fonts/"

# Add Montserrat fonts
for style in styles:
    font_style_name = style.capitalize()
    font_path = f"{font_dir}Montserrat-{font_style_name}.ttf"
    for size in sizes:
        font_name = f"montserrat-{style}{size}"  # e.g., montserrat-regular30
        fonts[font_name] = (font_path, size)

# Add Alkhemikal fonts
alkhemikal_path = f"{font_dir}Alkhemikal.ttf"
for size in sizes:
    font_name = f"Alkhemikal{size}"
    fonts[font_name] = (alkhemikal_path, size)
