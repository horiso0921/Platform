from model import Favot

class InteractFavot(Favot):
    def make_input(self, newspk, newutt, mode="normal", max_contexts=-1, id=None, idprefix="a"):
        return newutt[-512:]