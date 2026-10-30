import pytest

from framediff import tree_policy as policy


class CountedValues(list):
    def __iter__(self):
        self.calls=getattr(self,'calls',0)+1
        return super().__iter__()


def test_shared_provenance_is_loaded_and_aggregated_once(tmp_path,monkeypatch):
    groups=CountedValues(['train']);hashes=CountedValues(['train-sha'])
    source={'visual_geometry_contract':'v1','parser_training_groups':groups,
            'parser_training_hashes':hashes}
    reads=[]
    def read(path):reads.append(path);return source
    monkeypatch.setattr(policy,'read_json',read)
    paths=[str(tmp_path/'parser.json'),str(tmp_path/'nested/../parser.json')]
    rows=[{'predicted_target_elements':'prediction.json','parser_provenance':paths[i%2]} for i in range(100)]
    assert policy.detector_training_provenance(rows,'v1')==({'train'},{'train-sha'})
    assert len(reads)==groups.calls==hashes.calls==1


def test_inline_provenance_preserves_all_sources():
    groups=CountedValues(['train'])
    first={'visual_geometry_contract':'v1','parser_training_groups':groups}
    second={'visual_geometry_contract':'v1','parser_training_groups':['other'],
            'parser_training_hashes':['other-sha']}
    rows=[{'lazy_detector':{'sha':'x'},'parser_provenance':s} for s in (first,first,second)]
    assert policy.detector_training_provenance(rows,'v1')==({'train','other'},{'other-sha'})
    assert groups.calls==1


def test_every_row_still_requires_target_reference():
    source={'visual_geometry_contract':'v1'}
    rows=[{'predicted_target_elements':'prediction.json','parser_provenance':source},
          {'parser_provenance':source}]
    with pytest.raises(ValueError,match='Missing detector target cache'):
        policy.detector_training_provenance(rows,'v1')


@pytest.mark.parametrize('source,error',[(None,'Missing detector training provenance'),
                                      ({'visual_geometry_contract':'old'},'versions differ')])
def test_bad_provenance_rejected(source,error):
    rows=[{'predicted_target_elements':'prediction.json','parser_provenance':source}]
    with pytest.raises(ValueError,match=error):policy.detector_training_provenance(rows,'v1')
