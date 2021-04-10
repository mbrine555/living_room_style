import os
import shutil
import numpy as np

source = 'training'
dest = 'validation'
labels = os.listdir(source)

os.makedirs(dest, exist_ok=True)

for folder in labels:
	os.makedirs(os.path.join(dest, folder), exist_ok=True)
	
	files = os.listdir(os.path.join(source, folder))
	for file in files:
		if np.random.rand(1) < 0.1:
			shutil.move(os.path.join(source, folder, file), os.path.join(dest, folder, file))
	