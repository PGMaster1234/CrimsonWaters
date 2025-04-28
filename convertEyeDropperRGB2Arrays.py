def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    return [int(hex_value[i:i+2], 16) for i in (0, 2, 4)]

data = {
    "id": 4,
    "name": "OceanFront",
    "createdAt": 1745695443530,
    "sortBy": "m:asc",
    "colors": [
        {
            "hex": "#3b5f6f",
            "source": "ed",
            "captured": "2025-04-26T19:24:12.001Z"
        },
        {
            "hex": "#49787a",
            "source": "ed",
            "captured": "2025-04-26T19:24:17.232Z"
        },
        {
            "hex": "#568e8f",
            "source": "ed",
            "captured": "2025-04-26T19:24:24.537Z"
        },
        {
            "hex": "#94b6b4",
            "source": "ed",
            "captured": "2025-04-26T19:24:40.871Z"
        },
        {
            "hex": "#9f8f5b",
            "source": "ed",
            "captured": "2025-04-26T19:25:03.007Z"
        },
        {
            "hex": "#ac925f",
            "source": "ed",
            "captured": "2025-04-26T19:25:09.189Z"
        },
        {
            "hex": "#555f4f",
            "source": "ed",
            "captured": "2025-04-26T19:25:19.709Z"
        },
        {
            "hex": "#363b36",
            "source": "ed",
            "captured": "2025-04-26T19:25:27.699Z"
        },
        {
            "hex": "#5e7a7d",
            "source": "ed",
            "captured": "2025-04-26T19:26:12.425Z"
        },
        {
            "hex": "#43555f",
            "source": "ed",
            "captured": "2025-04-26T19:26:42.692Z"
        },
        {
            "hex": "#2f404a",
            "source": "ed",
            "captured": "2025-04-26T19:26:25.886Z"
        },
        {
            "hex": "#dcd8c9",
            "source": "ed",
            "captured": "2025-04-26T19:26:57.330Z"
        },
        {
            "hex": "#12161b",
            "source": "ed",
            "captured": "2025-04-26T19:26:48.358Z"
        },
        {
            "hex": "#9b6e53",
            "source": "ed",
            "captured": "2025-04-26T19:25:56.074Z"
        }
    ]
}

rgb_values = {}
for i, color_data in enumerate(data["colors"]):
    hex_value = color_data["hex"]
    rgb = hex_to_rgb(hex_value)
    color_name_parts = [part.capitalize() for part in hex_value[1:].split('f') if part]
    color_name = color_name_parts[0].lower() + "".join(color_name_parts[1:]) if color_name_parts else f"color{i+1}"
    rgb_values[color_name] = rgb

for name, rgb_list in rgb_values.items():
    print(f"{name} = {rgb_list}")