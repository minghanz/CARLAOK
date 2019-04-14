import time

class WorldClock:
    def __init__(self, world):
        self.time = 0
        self.lasttime = None
        world.on_tick(self.tick)

    def tick(self, timestamp):
        self.time = timestamp.elapsed_seconds
        #print(self.time)

    def dt(self, wait_for_zero=True):
        #print(self.time, self.lasttime, time.time())
        if self.lasttime == None:
            self.lasttime = self.time
            return 0.05
        else:
            ret = 0
            while ret == 0 and wait_for_zero:
                ret = self.time - self.lasttime
            self.lasttime = self.time
            return ret
        
