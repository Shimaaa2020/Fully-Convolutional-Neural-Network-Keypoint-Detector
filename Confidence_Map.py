import numpy as np

def confidence_map(X, gt):
    a, b, c, d = X.shape #a = number of samples, b = image_row, c = image_column, d = number of key points
    sigma = 10
    mask = np.zeros([a, b, c, d])
    conf_map = np.zeros([a, b, c, d])
    for i in range(a):
        for k in range(d):
            center_x = int(gt[i, 2 * k])
            center_y = int(gt[i, 2 * k + 1])
            for y in range(b):
                for x in range(c):
                    temp = np.exp((-(pow(x - center_x, 2) + pow(y - center_y, 2)))/sigma)
                    if temp < 0.1:
                        mask[i, y, x, k] = 0
                    else:
                        mask[i, y, x, k] = 1
                    conf_map[i, y, x, k] = temp

    return mask, conf_map
