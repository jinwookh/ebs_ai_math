import glob
import os
from PIL import Image
import numpy as np
import pandas as pd

images = []
for f in glob.glob("test_jinwookh_data/*"):
	images.append(Image.open(f))



r = 0.2989
g = 0.5870
b = 0.1140

for image in images:
	print(image.filename)


gray_nps = []
for image in images:
	resized = image.resize((28,28))
	resized_array = np.asarray(resized)
	gray = resized_array[:,:,0] * r + resized_array[:,:,1] *g + resized_array[:,:,2] * b	
	gray = gray.reshape(1,784)
	gray = gray.astype(np.uint8)
	gray_nps.append(gray)


csv_store_file_path = "data/jinwookh_test.csv"
os.remove(csv_store_file_path)

for gray_np in gray_nps:
	pd.DataFrame(gray_np).to_csv(csv_store_file_path, mode="a")

