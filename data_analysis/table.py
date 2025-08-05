import pandas as pd

# file to make csv
data = {
    'video_id': [],         
    'model': [],            
    'LitterNET_used': [],
    'litter_present': [],
    'litter_detected': [],   
    'accuracy': [],
    'confidence': []
}

# id = even, trash is present
# odd = no trash                                                                                                                                                

df = pd.DataFrame(data)
with open("results.txt", 'r') as f:
    #f.write(filename + " ")
    f.read()
    
