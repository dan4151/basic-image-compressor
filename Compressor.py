from PIL import Image
import numpy as np

def k_aprox(matrix, k):
    s, v, d = np.linalg.svd(matrix, full_matrices=False)
    approx = np.matrix(s[:, :k]) * np.diag(v[:k]) * np.matrix(d[:k, :])
    return approx

k = 5
image = Image.open("mcaffe.jpg")
pix = np.array(image)
r = pix[:,:,0]
g = pix[:,:,1]
b = pix[:,:,2]
r_k = k_aprox(r, k)
g_k = k_aprox(g, k)
b_k = k_aprox(b, k)
pix[:,:,0] = r_k
pix[:,:,1] = g_k
pix[:,:,2] = b_k
pil_image=Image.fromarray(pix)
pil_image.show()