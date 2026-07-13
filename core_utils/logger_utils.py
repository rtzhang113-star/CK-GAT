import time


class Logger:
    def __init__(self, args):
        self.args = args

    def log(self, string):
        if string[0] == "\n":
            print("\n", end="")
            string = string[1:]
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time())), string)

    def __call__(self, string):
        if self.args.verbose:
            self.log(string)

    def print(self, string):
        self.args.verbose = 1
        self.__call__(string)
        self.args.verbose = 0
