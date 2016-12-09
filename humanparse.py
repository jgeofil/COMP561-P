f = open("data/data.geno", "rb")
try:
    byte = f.read(10)
    for i in range(3000):
        print byte
        # Do stuff with byte.
        byte = f.read(10)
finally:
    f.close()
