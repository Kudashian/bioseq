class Sequencer:
    def __init__(self, pid, sequence, refgenome, organism, technology):
        self.pid = pid
        self.sequence = sequence
        self.refgenome = refgenome
        self.organism = organism
        self.technology = technology
    
    def __str__(self):
        return f"Patient ID: {self.pid} /n Length: {len(self.sequence)} bp /n Organism: {self.organism}"
    
    def __len__(self):
        return len(self.sequence)
    
    def __repr__(self):
        seq_preview = self.sequence[:25]
        return f"Sequencer(pid = {self.pid!r}, sequence = {seq_preview}, refgenome = {self.refgenome}, organism = {self.organism}, technology = {self.technology})"