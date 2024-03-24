class Op:
    def __init__(self):
        pass

    def profile(self, batch_data):
        """Do some transformation"""
        NotImplementedError()

    def get_result(self):
        NotImplementedError()
    