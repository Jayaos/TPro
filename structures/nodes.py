import numpy as np

class ConstitutiveNode:
    '''
    Highest-level parent nodes which
    defined constitutive expression
    of TranscriptionFactor output
    '''
    def __init__(self, output):
        '''
        Attributes
        ----------
        output (TranscriptionFactor) : the protein which is
            constitutively expressed
        '''
        self.output = output
        
    def evaluate(self, ligands):
        return [self.output]
        
    def __str__(self):
        return "<*, <" + self.output.__str__() + ">>"


class TPNode:
    '''
    Nodes which manage transport of proteins
    between Cassettes in a transcriptional program.
    '''
    def __init__(self, program, parents):
        '''
        Attributes
        ----------
        program (TranscriptionalProgram) : the program which
            regulates expression of proteins through this Node
        parents (list of Node) : the Nodes which feed proteins
            to this node from earlier phases in the program
        target_protein (TranscriptionFactor) or (list of TranscriptionFactor) :
            the protein(s) whose expression is regulated by the program in this Node
        '''
        self.program = program
        self.parents = parents
        self.target_protein = self.program.output
    
    def evaluate(self, ligands):
        '''
        yield output based on presence (1) 
        of ligands or lack thereof (0) by
        obtaining proteins (TFs) from higher
        layers of the graph
        
        Parameters
        ----------
        ligands (list of str) : list of ligands present
        
        Return
        ------
        proteins (list of TranscriptionFactor) : contains
            TranscriptionFactors present in the program after
            evaluating this node, which also includes all proteins
            from earlier phases of the program
        '''
        proteins = []
        for parent in self.parents:
            proteins += parent.evaluate(ligands)
        
        program_out = self.program.evaluate(proteins, ligands)
        if isinstance(program_out, list):
            for i in range(len(program_out)):
                if program_out[i]:
                    proteins.append(self.program.output[i])
        else:
            if program_out:
                proteins.append(self.program.output)
        
        return proteins
    
    def logical_output(self, ligands):
        '''
        same functionality as .evaluate(), but
        returns integer 1 if the target_protein
        is present after evaluation of this node
        
        Parameters
        ----------
        ligands (list of str) : list of ligands present
        
        Return
        ------
        logic_output (int) : is 1 if all protein(s) in
            target_protein are expressed after evaluation
            of the program in this node
        '''
        proteins = self.evaluate(ligands)
        
        if isinstance(self.target_protein, list):
            outs = []
            for protein in self.target_protein:
                if protein in proteins:
                    outs.append(1)
                else:
                    outs.append(0)
            return np.bitwise_and.reduce(outs, axis=0)
        else:
            if self.target_protein in proteins:
                return 1
            else:
                return 0
    
    def __str__(self):
        return self.program.__str__()