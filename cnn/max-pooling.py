X = [
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [1, 2, 0, 1],
    [3, 1, 2, 4]
]

def maxpool2x2(mat):
    out = []
    for i in range(0, len(mat), 2):
        row = []
        for j in range(0, len(mat[0]), 2):
            block = [
                mat[i][j], mat[i][j+1],
                mat[i+1][j], mat[i+1][j+1]
            ]
            row.append(max(block))
        out.append(row)
    return out

output = maxpool2x2(X)

print("Pooled Output:")
for row in output:
    print(row)