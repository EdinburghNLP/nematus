#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess, threading
from scorer import Scorer
from reference import Reference

class BeerError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class BeerScorer(Scorer):
    """
    Python wrapper for the BEER metric. Starts a BEER process and keeps it alive, so that the model
    can be kept in memeory. Arguments are the BEER language abbreviation and the path to the BEER
    installation. They need to be specified as follows:"beer_language=lg,beer_path=path" (any order).
    """
    def __init__(self, argument_string):
        Scorer.__init__(self, argument_string)
        
        #Lock for the BEER process, which can only handle one request at a time:
        self.lock = threading.Lock()
        
        #Get necessary arguments for starting BEER from argument string parsed in Scorer.__init__()
        self._beer_language = self._arguments["beer_language"]
        self._beer_path = self._arguments["beer_path"] + "/"
        
        #Start a BEER process:
        command = self._beer_path+"beer -l "+self._beer_language+" --workingMode interactive "
        self.beer_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def set_reference(self, reference_tokens):
        """
        Construct a BeerReference from a sequence of tokens and make it the reference against which the scorer evaluates hypotheses.
        This can be done any time.
        """
        self.lock.acquire()
        self._reference = BeerReference(reference_tokens, self)
        self.lock.release()

    def terminate_process(self):
        """
        Waits for the current request to be processed and terminates the BEER process.
        """
        self.lock.acquire()
        self.beer_process.terminate()
        self.lock.release()
        
    def kill_process(self):
        """
        Kills the BEER process right away.
        """
        self.beer_process.kill()

class BeerReference(Reference):
    """
    BEER reference object, against which hypotheses can be scored.
    """
    def __init__(self, reference_tokens, beer_scorer):
        Reference.__init__(self, reference_tokens)
        
        #Construct reference string from tokens
        self._reference_string = " ".join(reference_tokens)
        self._beer_scorer = beer_scorer

    def score(self, hypothesis_tokens):
        
        #Construct hypothesis string from hypothesis tokens:
        hypothesis_string = " ".join(hypothesis_tokens)
        
        #Acquire lock to make sure BEER process is not in use:
        self._beer_scorer.lock.acquire()
        
        #Score hypothesis string against reference string
        try:
            self._beer_scorer.beer_process.stdin.write("EVAL ||| "+hypothesis_string+" ||| "+self._reference_string+"\n")
        except:
            raise BeerError("Beer returned the following error: "+ self._beer_scorer.beer_process.stderr.readline().strip())
        
        #Read feature values from process output
        std_out = self._beer_scorer.beer_process.stdout.readline()
        #Release the process lock
        self._beer_scorer.lock.release()
        
        #Check if BEER returned a score:
        try:
            n = float(std_out)
        except:
            raise BeerError("Beer returned the following error: "+ self._beer_scorer.beer_process.stderr.readline().strip())
        #Return final score
        return n