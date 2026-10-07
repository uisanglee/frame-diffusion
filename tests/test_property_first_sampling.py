import random
from collections import Counter

import pytest

from framediff.tree_online import choose_mutation_site, sample_mutation
from framediff.tree_edits import FIELDS


def test_property_first_is_not_weighted_by_owner_count():
    sites=[(i,'width','100px','') for i in range(1,21)]
    sites += [(21,'row-gap','20px','')]
    rng=random.Random(42)
    counts=Counter(choose_mutation_site(sites,rng)[1] for _ in range(20000))
    assert .48 < counts['width']/20000 < .52


def test_cached_witness_count_does_not_bias_owner_selection():
    sites=[(1,'width',f'{i}px','') for i in range(20)]
    sites += [(2,'width','100px',''),(3,'row-gap','20px','')]
    rng=random.Random(43)
    counts=Counter(choose_mutation_site(sites,rng)[:2] for _ in range(20000))
    assert .23 < counts[(1,'width')]/20000 < .27
    assert .23 < counts[(2,'width')]/20000 < .27
    assert .48 < counts[(3,'row-gap')]/20000 < .52


def test_mutation_uses_property_first_and_preserves_priority():
    state=[[['',''] for _ in FIELDS] for _ in range(22)]
    for i in range(1,21):state[i][FIELDS.index('width')]=['100px','important']
    state[21][FIELDS.index('row-gap')]=['20px','']
    rng=random.Random(44)
    edits=[sample_mutation(state,[500,300],rng) for _ in range(10000)]
    assert .47 < sum(e[1]=='width' for e in edits)/len(edits) < .53
    assert all(e[3]==('important' if e[1]=='width' else '') for e in edits)


def test_single_site_probe_rng_and_empty_candidates():
    site=(1,'width','100px','')
    left=random.Random(4);right=random.Random(4)
    assert choose_mutation_site([site],left)==right.choice([site])
    assert left.getstate()==right.getstate()
    with pytest.raises(ValueError,match='No supported existing'):
        choose_mutation_site([],left)
