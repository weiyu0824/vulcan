import yaml

# Load YAML data from a file
with open('video_analytics/config.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Print the loaded YAML data
print(data)