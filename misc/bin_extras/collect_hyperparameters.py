import os
import pandas as pd
import re

directory_path = "/home/essharom/code/cancer-gnn-nf/results/hyperparameters"

output_file = "/home/essharom/code/cancer-gnn-nf/results/hyperparameters.csv"



# Function to extract the model name and dataset name from the file name
def extract_model_dataset_names(filename):
    parts = filename.split("_")
    model_name = parts[2]
    dataset_name = "_".join(parts[3:]).split(".")[0]
    return model_name, dataset_name

# Function to handle special characters in numerical values
def handle_special_characters(value):
    numeric_value = re.findall(r'[+-]?\d+(?:\.\d+)?', value)
    if numeric_value:
        return float(numeric_value[0])
    else:
        return value

# Path to the directory containing the best trial files
# directory_path = "/path/to/best_trial_files_directory"

# Initialize an empty list to store the data
data = []

# Loop through the files in the directory
for filename in os.listdir(directory_path):
    if filename.startswith("best_trial"):
        model_name, dataset_name = extract_model_dataset_names(filename)
        with open(os.path.join(directory_path, filename), 'r') as file:
            lines = file.readlines()
            # Extract the value using regular expressions to handle special characters
            bacc_value = handle_special_characters(re.findall(r'[+-]?\d+(?:\.\d+)?', lines[1])[0])
            params = {}
            for line in lines[3:]:
                key, value = line.split(":")
                params[key.strip()] = handle_special_characters(value.strip())
            data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'BACC': bacc_value,
                **params
            })

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Save the data table to a CSV file
#output_file = "/path/to/output_file.csv"
df.to_csv(output_file, index=False)

print("Data table has been saved to:", output_file)
