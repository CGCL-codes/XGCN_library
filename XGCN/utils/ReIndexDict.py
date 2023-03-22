class ReIndexDict:
    
    def __init__(self):
        self.d = {}
        self.rev_d = None
        self.cnt = 0
        
    def __getitem__(self, old_id):
        if old_id in self.d:
            return self.d[old_id]
        else:
            new_id = self.cnt
            self.d[old_id] = new_id
            self.cnt += 1
            return new_id
        
    def get_old2new_dict(self):
        return self.d
    
    def get_new2old_dict(self):
        if self.rev_d is None:
            self.rev_d = {}
            for old_id in self.d:
                self.rev_d[self.d[old_id]] = old_id
        return self.rev_d
