import numpy as np

class SEPAProgram:
    '''
    Defines logical behavior of a Cassette
    which can take arbitrarily many input TFs
    to regulate expression of a single output.
    '''
    def __init__(self, transcription_factors, output):
        '''
        All TFs must have the same DNA binding domain

        Attributes
        ----------
        transcription_factors (list of TranscriptionFactor) : the TFs
            which bind to the promoter of the cassette
        output (TranscriptionFactor) : the protein whose expression is regulated by
            TFs binding to the promoter of the cassette
        dbd (str) : three character string indicating DNA binding
            domain of TF
        n_factors (int) : number of TFs that can bind to the promoter
        '''
        self.transcription_factors = transcription_factors
        self.output = output
        
        self.dbd = self.transcription_factors[0].dbd
        self.n_factors = len(self.transcription_factors)
        
        assert np.array([self.transcription_factors[i].dbd == self.dbd for i in range(self.n_factors)]).all()
    
    def evaluate(self, proteins, ligands):
        '''
        yield output based on presence (1) 
        of proteins and ligands or lack thereof (0)
        
        Parameters
        ----------
        proteins (list) : list of transcription factors present
        ligands (list of str) : list of ligands present
        
        Return
        ------
        output (int) : 1 indicates expression
        '''
        outs = []
        for transcription_factor in self.transcription_factors:
            if transcription_factor in proteins:
                outs.append(transcription_factor.evaluate(ligands))
            else:
                outs.append(1)
        return np.bitwise_and.reduce(outs, axis=0)
    
    def __str__(self):
        tup_str = "<{}, ".format(self.dbd)
        for transcription_factor in self.transcription_factors:
            tup_str += ",".join(transcription_factor.__str__().split(",")[1:]) + ", "
        if self.output.dbd == "GFP":
            tup_str += "<GFP>>"
        elif self.output.dbd == "O":
            tup_str += "<O>>"
        else:
            tup_str += "<" + self.output.__str__() + ">>"
        return tup_str


class MIMOProgram:
    '''
    Defines logical behavior of a set of Cassettes
    which each take one input TF to regulate expression
    of one output.
    '''
    def __init__(self, transcription_factors, output):
        '''
        The number of TFs in input should be the same as
        that of output

        Attributes
        ----------
        transcription_factors (list of TranscriptionFactor) : the TFs
            which bind to the promoters of the cassettes
        output (list of TranscriptionFactor) : the proteins whose expression
            is regulated by TFs binding to the promoters of the cassettes
        n_factors (int) : number of cassettes represented by this program
        '''
        self.transcription_factors = transcription_factors
        self.output = output
        
        self.n_factors = len(self.transcription_factors)
        
        assert self.n_factors == len(self.output)
    
    def evaluate(self, proteins, ligands):
        '''
        yield output based on presence (1) 
        of proteins and ligands or lack thereof (0)
        
        Parameters
        ----------
        proteins (list) : list of transcription factors present
        ligands (list of str) : list of ligands present
        
        Return
        ------
        output (list of int) : 1 indicates expression for each output
        '''
        outs = []
        for transcription_factor in self.transcription_factors:
            if transcription_factor in proteins:
                outs.append(transcription_factor.evaluate(ligands))
            else:
                outs.append(1)
        return outs
    
    def __str__(self):
        tup_str = "<"
        for i in range(self.n_factors):
            if i != 0:
                tup_str += ", "
            tup_str += self.transcription_factors[i].__str__() + ", "
            if self.output[i].dbd == "GFP":
                tup_str += "<GFP>"
            elif self.output[i].dbd == "O":
                tup_str += "<O>>"
            else:
                tup_str += "<" + self.output[i].__str__() + ">"
        tup_str += ">"
        return tup_str