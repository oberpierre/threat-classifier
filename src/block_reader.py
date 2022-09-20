def blocks(infile, bufsize=1024*1024):
    while True:
        try:
            data = infile.readlines(bufsize)
            if data:
                yield data
            else:
                break
        except IOError as err:
            print("I/O error({}): {}".format(err[0], err[1]))
            break
