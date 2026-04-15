
X_edge = [
    [10,10,10],
    [0, 0, 0],
    [10,10,10]
]

# horizontal edge kernel
K_edge = [
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
]

def conv2d_3x3(img, kernel):
    out = []
    for i in range(len(img)-2):
        row = []
        for j in range(len(img[0])-2):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    s += img[i+ki][j+kj] * kernel[ki][kj]
            row.append(s)
        out.append(row)
    return out

edge_output = conv2d_3x3(X_edge, K_edge)
print("Edge Detection Output:", edge_output)

#here no edge for this, no directional edge tho