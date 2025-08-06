import pandas as pd

# Read the CSV with NO headers
df = pd.read_csv("results.csv", header=None)
start = 300 # starting_frame
end = 305 # ending_frame for pattern
# Create condition mask
df[3] = pd.to_numeric(df[3], errors='coerce')
# change 003 to current video type
mask = (df[0] == "008") &  (df[1] == "RF-DETRs") & (df[4] == 2) & (df[3].between(start, end, inclusive='both'))

# Apply the change
df.loc[mask, 5] = True

# Save the modified CSV without headers and index
df.to_csv("results.csv", index=False, header=False)