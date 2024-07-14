
class TranscriptionFactor:
    '''
    Most basic logical unit, acts as either
    a BUFFER (repression == "+" or "S") or
    as NOT (repression == "A").
    '''
    def __init__(self, dbd, rcd, repression):
        '''
        Attributes
        ----------
        dbd (str) : three character string indicating
            DNA binding domain of TF
        rcd (str) : string indicator of regulatory
            core domain (defining ligand attachment)
        repression (str) : indicator for TF behavior,
            must be + (repressor), A (anti-repressor),
            or A (super-repressor)
        '''
        self.dbd = dbd
        self.rcd = rcd
        self.repression = repression
        
        assert self.repression in ["+", "A", "S", ""]
        
    def evaluate(self, ligands):
        '''
        yield output based on presence (1) 
        of ligands or lack thereof (0)
        
        Parameters
        ----------
        ligands (list of str) : list of ligands present
        
        Return
        ------
        output (int) : 1 indicates expression, 0 otherwise
        '''
        if self.repression == "+":
            return int(self.rcd in ligands)
        elif self.repression == "A":
            return int(not (self.rcd in ligands))
        elif self.repression == "S":
            return 0
        
    def __eq__(self, tf):
        if isinstance(tf, TranscriptionFactor):
            return (self.dbd == tf.dbd) and (self.rcd == tf.rcd) and (self.repression == tf.repression)
        return False
        
    def __str__(self):
        return "{}, {}, {}".format(self.dbd, self.rcd, self.repression)


class Cassette:
    '''
    Utility class for defining building
    blocks of a transcriptional program.
    Represents single-input promotor and
    expression of protein.
    '''
    def __init__(self, promoter, transcription_factor):
        '''
        Attributes
        ----------
        promoter (str) : three character string indicating
            DNA binding domain of TF or 'constitutive'
        transcription_factor (TranscriptionFactor) : the protein
            whose expression is regulated by TFs binding to the
            promoter of the cassette
        '''
        self.promoter = promoter # three-character dbd or 'constitutive'
        self.transcription_factor = transcription_factor