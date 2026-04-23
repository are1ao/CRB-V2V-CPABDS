from imm_core import IMM

class IMMManager:
    def __init__(self):
        self.filters = {}

    def get_filter(self, vid):
        if vid not in self.filters:
            self.filters[vid] = IMM()
        return self.filters[vid]

    def step(self, vid, z):
        imm = self.get_filter(vid)
        return imm.step(z)