#!/usr/bin/env python

import subprocess, threading
from metrics.scorer import Scorer
from metrics.reference import Reference

class MeteorError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MeteorScorer(Scorer):
    """
    Python wrapper for the METEOR metric. Starts a METEOR process and keeps it alive, so that the model
    can be kept in memeory. Arguments are the meteor language abbreviation and the path to the METEOR
    installation. They need to be specified as follows:"meteor_language=lg,meteor_path=path" (any order).
    """
    def __init__(self, argument_string):
        Scorer.__init__(self, argument_string)
        
        #Lock for the METEOR process, which can only handle one request at a time:
        self.lock = threading.Lock()
        
        #Get necessary arguments for starting METEOR from argument string parsed in Scorer.__init__()
        self._meteor_language = self._arguments["meteor_language"]
        self._meteor_path = self._arguments["meteor_path"] + "/"
        
        #Start a METEOR process:
        command = "java -Xmx2G -jar "+self._meteor_path+"meteor-*.jar - - -l "+self._meteor_language+" -stdio"
        self.meteor_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def set_reference(self, reference_tokens):
        """
        Construct a MeteorReference from a sequence of tokens and make it the reference against which the scorer evaluates hypotheses.
        This can be done any time.
        """
        self.lock.acquire()
        self._reference = MeteorReference(reference_tokens, self)
        self.lock.release()

    def terminate_process(self):
        """
        Waits for the current request to be processed and terminates the METEOR process.
        """
        self.lock.acquire()
        self.meteor_process.terminate()
        self.lock.release()
        
    def kill_process(self):
        """
        Kills the METEOR process right away.
        """
        self.meteor_process.kill()

class MeteorReference(Reference):
    """
    METEOR reference object, against which hypotheses can be scored.
    """
    def __init__(self, reference_tokens, meteor_scorer):
        Reference.__init__(self, reference_tokens)
        
        #Construct reference string from tokens
        self._reference_string = " ".join(reference_tokens)
        self._meteor_scorer = meteor_scorer

    def score(self, hypothesis_tokens):
        
        #Construct hypothesis string from hypothesis tokens:
        hypothesis_string = " ".join(hypothesis_tokens)
        
        #Acquire lock to make sure METEOR process is not in use:
        self._meteor_scorer.lock.acquire()
        
        #Score hypothesis string against reference string
        try:
            self._meteor_scorer.meteor_process.stdin.write("SCORE ||| "+self._reference_string+" ||| "+hypothesis_string+"\n")
        except:
            raise MeteorError("Meteor returned the following error: "+ self._meteor_scorer.meteor_process.stderr.readline().strip())
        
        #Read feature values from process output
        std_out = self._meteor_scorer.meteor_process.stdout.readline()
        
        #Pass feature values to METEOR process for computation of the final score
        try:
            self._meteor_scorer.meteor_process.stdin.write("EVAL ||| "+std_out)
        except:
            raise MeteorError("Meteor returned the following error: "+ self._meteor_scorer.meteor_process.stderr.readline().strip())
        std_out = self._meteor_scorer.meteor_process.stdout.readline()
        
        #Release the process lock
        self._meteor_scorer.lock.release()
        
        #Check if Meteor returned a score:
        try:
            n = float(std_out)
        except:
            raise MeteorError("Meteor returned the following error: "+ self._meteor_scorer.meteor_process.stderr.readline().strip())
        
        #Return final score
        return n
