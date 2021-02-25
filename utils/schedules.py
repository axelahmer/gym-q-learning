class LinearSchedule(object):
    def __init__(self, begin, end, nsteps):
        self.value = begin
        self.begin = begin
        self.end = end
        self.nsteps = nsteps

    def update(self, t):
        """
        sets and returns the interpolation value from step t
        """
        assert t >= 0
        if t < self.nsteps:
            self.value = self.begin + t * (self.end - self.begin) / self.nsteps
        else:
            self.value = self.end

        return self.value

    def value(self):
        return self.value
