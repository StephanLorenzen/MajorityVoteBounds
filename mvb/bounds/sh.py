#
# Implementation of single hypothesis bound
#
from .pbkl import PBkl

def SH(empirical_risk, m, delta=0.05):
    return 0.5*PBkl(empirical_risk, m, KLQP=0.0, delta=delta)
