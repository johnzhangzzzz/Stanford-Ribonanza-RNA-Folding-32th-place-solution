class BPPS():
    def __init__(self):
        pass
    def countbpps(self,seq,dev=device):
        aass=bpps(seq, package="eternafold")
        aass=torch.tensor(aass,dtype=torch.float16)
        return aass