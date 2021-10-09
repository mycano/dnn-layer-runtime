

def get_size(width, kernel_size, stride, padding=0):
    out = width - kernel_size + 2 * padding
    out = out / stride + 1
    return out

if __name__ == '__main__':
    print(get_size(224, 11, 4))