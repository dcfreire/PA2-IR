from multiprocessing import Manager, Process


class Logger:
    def __init__(self):
        m = Manager()
        self.queue = m.Queue()
        self.start()

    def start(self):
        Process(target=self.worker).start()

    def worker(self):
        while True:
            msg = self.queue.get()
            if msg == "shutdown":
                return
            print(msg)

    def add_message(self, msg: str):
        self.queue.put(msg)

    def shutdown(self):
        self.queue.put("shutdown")
