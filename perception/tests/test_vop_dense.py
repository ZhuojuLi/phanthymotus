import numpy as np
import pytest
from plugins.vision_runtime import decode_detections, LetterboxMeta, VisionDecodeError
META={'detection_layout':'yolo_dense_v1','num_classes':220,'nms_iou':.7,'max_det':300}
def test_dense_class_aware_nms_before_clipping_and_threshold():
    x=np.zeros((1,224,8400),np.float32)
    x[0,:4,:4]=np.array([[100,100,100,300],[100,100,100,300],[80,80,80,20],[80,80,80,20]])
    x[0,4,:4]=[.9,.8,.2,.34];x[0,5,2]=.85
    b,s,c=decode_detections([x],LetterboxMeta(.5,20,10,1280,720),.35,metadata=META)
    assert list(c)==[0,1]  # same-class duplicate removed, cross-class overlap retained
    np.testing.assert_allclose(s,[.9,.85]);np.testing.assert_allclose(b,[[80,100,240,260]]*2)
def test_dense_empty_invalid_shape_and_nonfinite():
    x=np.zeros((1,224,8400),np.float16);m=LetterboxMeta(1,0,0,640,640)
    assert decode_detections(x,m,.35,metadata=META)[0].shape==(0,4)
    with pytest.raises(VisionDecodeError):decode_detections(x[:,:,:10],m,.35,metadata=META)
    x[0,0,0]=np.nan
    with pytest.raises(VisionDecodeError):decode_detections(x,m,.35,metadata=META)
def test_legacy_detection_not_changed():
    x=np.array([[[10,20,30,40,.9,2],[10,20,30,40,.8,2]]],np.float32)
    b,s,c=decode_detections(x,LetterboxMeta(1,0,0,640,640),.35)
    assert len(b)==2 and list(c)==[2,2]
