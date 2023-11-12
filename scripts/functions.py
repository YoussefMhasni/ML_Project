import numpy as np
def encoder(filename):
  from PIL import Image
  image=Image.open(filename).resize((32,32)).convert('RGB')
  image=np.array(image)
  vect= np.concatenate((image[:,:,0].ravel(), image[:,:,1].ravel(),image[:,:,2].ravel()))
  return vect[None,:]