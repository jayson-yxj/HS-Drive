import json
import numpy as np
import pytest
from tools.m8j8_report import paired, write


def example(scores, x=None):
    gt = np.array([[0.,0.,0.,1.,1.,2.,0.,0.,0.], [8.,0.,0.,1.,1.,2.,0.,0.,0.]])
    boxes = gt.copy()
    if x is not None:
        boxes[:,0] = x
    return dict(boxes=boxes,scores=np.array(scores),labels=np.array([0,0]),gt=gt,gt_labels=np.array([0,0]))


def test_paired_errors_use_only_the_same_detected_GT():
    a,b = example([.9,.9]),example([.9,.1],[.5,8.])
    result = paired([a],[b],['car','absent'])
    assert (result['common'],result['A_only'],result['B_only'],result['neither']) == (1,1,0,0)
    assert result['paired_errors']['xy_m_B_minus_A']['mean'] == .5
    assert result['paired_errors']['xy_m_B']['count'] == 1
    assert result['per_class_counts']['absent'] == dict(common=0,A_only=0,B_only=0,neither=0)


def test_disjoint_detections_do_not_fabricate_zero_paired_error():
    result = paired([example([.9,.1])],[example([.1,.9])],['car'])
    assert (result['common'],result['A_only'],result['B_only'],result['neither']) == (0,1,1,0)
    assert result['paired_errors'] == {}


def test_numpy_aggregate_serialization_and_existing_evidence_protection(tmp_path):
    path = tmp_path/'report.json'
    write(path,dict(count=np.int64(2),mean=np.float32(.5)))
    assert json.loads(path.read_text()) == dict(count=2,mean=.5)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write(path,dict(count=99))
    assert path.read_bytes() == before
