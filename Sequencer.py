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
    
    def __getattribute__(self, attr_name):
        return super().__getattribute__(attr_name)
    
    def __setattr__(self, attr_name, value):
        super().__setattr__(attr_name, value)

    def __contains__(self, sub_seq):
        return sub_seq in self.sequence
    
    def __dir__(self):
        return super().__dir__() + ['pid', 'sequence', 'refgenome', 'organism', 'technology']
    
    def __getitem__(self, index):
        return self.sequence[index]
    
    def __eq__(self, other):
        if not isinstance(other, Sequencer):
            return False
        return self.sequence == other.sequence
    
    '''GC content of the sequence'''
    @property 
    def gc_content(self):
        self.sequence = self.sequence.lower()
        gc_count = self.sequence.count('g') + self.sequence.count('c')
        return (gc_count / len(self.sequence)) * 100 if len(self.sequence) > 0 else 0
    
    '''CpG sites in the sequence'''    
    @property 
    def cpg_sites(self):
        return self.sequence.count('cg')
    
    '''AT content of the sequence'''
    @property 
    def at_content(self):
        self.sequence = self.sequence.lower()
        at_count = self.sequence.count('a') + self.sequence.count('t')
        return (at_count / len(self.sequence)) * 100 if len(self.sequence) > 0 else 0
    
    '''Complementary DNA strand'''
    @property 
    def complementary_strand(self):
        complement = {'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
        return ''.join(complement[base] for base in self.sequence.lower())
    