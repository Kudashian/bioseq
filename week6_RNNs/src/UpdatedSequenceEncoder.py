import torch
import torch.nn as nn

class SequenceEncoder:
    """
    Converts DNA sequences to one-hot encoded tensors.
    """
    
    def __init__(self):
        self.nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.index_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    def encode(self, sequence: str) -> torch.Tensor:
        """
        Convert DNA sequence to one-hot tensor.
        """
        one_hot = torch.zeros((len(sequence), 4), dtype=torch.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide.upper() in self.nucleotide_to_index:
                index = self.nucleotide_to_index[nucleotide.upper()]
                one_hot[i, index] = 1.0
            else:
                '''Handle unknown nucleotides (e.g., 'N') by leaving the row as zeros.'''
                pass
        return one_hot
    
    def decode(self, tensor: torch.Tensor) -> str:
        """
        Convert one-hot tensor back to DNA sequence.
        For validation/debugging.
        """
        sequence = []
        for i in range(tensor.shape[0]):
            if torch.sum(tensor[i]) == 0:
                '''Handle unknown nucleotides (e.g., 'N') by adding 'N' to the sequence.'''
                sequence.append('N')
                continue
            index = torch.argmax(tensor[i]).item()
            nucleotide = self.index_to_nucleotide[index]
            sequence.append(nucleotide)

        return ''.join(sequence)
    
