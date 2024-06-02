class ProcessOp():
    def __init__(self):
        self.compute_latencies = [] 
        self.input_size = None

    def profile(self, batch_data):
        """Do some transformation"""
        NotImplementedError()

    def get_compute_latency(self):
        if len(self.compute_latencies) == 0:
            return 0
        return sum(self.compute_latencies)/len(self.compute_latencies)

    def get_input_size(self):
        return self.input_size 

    def get_result(self):
        pass

class SourceOp():
    def load_batch(self):
        """Do some transformation"""
        NotImplementedError() 