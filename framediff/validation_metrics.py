"""Cheap declaration diagnostics; never change training labels or render HTML."""
from decimal import Decimal
import re

from .tree_edits import UNITS

CONTRACT = 'css-action-diagnostics-numeric-v1'
NUMBER = re.compile(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))([a-z%]*)')


def canonical_value(value):
    """Equal numeric spellings in the same unit, without layout-dependent conversion."""
    match=NUMBER.fullmatch(value)
    if match:
        number,unit=match.groups();number=Decimal(number)
        if unit in UNITS:
            return ('number',number,unit)
        # CSSOM serializes a unitless length zero as 0px for these six fields.
        # In particular, do not equate 0% with 0px (indefinite percentage sizes).
        if not unit and number==0:
            return ('number',number,'px')
    return ('literal',value)


def declaration_equal(left,right):
    if left==right:return True
    return left[1]==right[1] and canonical_value(left[0])==canonical_value(right[0])


def differing_sites(current,target,fields):
    if len(current)!=len(target) or any(len(row)!=len(fields) for row in (*current,*target)):
        raise ValueError('Target declaration state shape changed')
    return {(owner,field) for owner in range(1,len(current)) for j,field in enumerate(fields)
            if not declaration_equal(current[owner][j],target[owner][j])}
