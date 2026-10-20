import pytest
import pandas as pd
from src.detectors import RollingZScoreDetector
@pytest.mark.parametrize('window',[0,1,-1,True,2.5,'5',None])
def test_invalid_windows_fail_before_scoring(window):
 with pytest.raises(ValueError,match='window'):
  RollingZScoreDetector(window=window)
@pytest.mark.parametrize('threshold',[0,-1,float('nan'),float('inf'),True,'3',None])
def test_invalid_thresholds_fail_before_scoring(threshold):
 with pytest.raises(ValueError,match='threshold'):
  RollingZScoreDetector(threshold=threshold)
def test_smallest_window_detects_spikes():
 data=pd.DataFrame({'latency_ms':[10,10,100],'error_rate':[0,0,0]})
 assert RollingZScoreDetector(window=2,threshold=3).predict(data).tolist()==[False,False,True]
