import pickle

# Load the trajectories from the file
with open("cartpole_trajectories.pkl", "rb") as f:
    loaded_trajectories = pickle.load(f)

# Check the loaded data
print(f"Loaded {len(loaded_trajectories)} trajectories.")

# Optionally, inspect the first trajectory
print("First trajectory data:", loaded_trajectories[0])
