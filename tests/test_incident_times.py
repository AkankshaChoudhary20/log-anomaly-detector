import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate import group_into_incidents


def frame(times,index):
    return pd.DataFrame({'timestamp':times,'latency_ms':[10,99,20][:len(times)],'error_rate':[.1,.5,.2][:len(times)]},index=index)


def test_adjacent_rows_separated_in_time_are_separate_incidents():
    data=frame(['2026-01-01T00:00:00Z','2026-01-01T00:10:00Z'],[0,1])
    incidents=group_into_incidents(data,pd.Series([True,True],index=data.index))
    assert len(incidents)==2
    assert all(item['duration_minutes']==1 for item in incidents)


def test_nonconsecutive_indices_use_time_and_include_intermediate_peaks():
    data=frame(['2026-01-01T00:00:00Z','2026-01-01T00:02:00Z','2026-01-01T00:04:00Z'],[10,90,1000])
    incidents=group_into_incidents(data,pd.Series([True,False,True],index=data.index))
    assert len(incidents)==1
    assert incidents[0]['duration_minutes']==5
    assert incidents[0]['peak_latency_ms']==99


def test_unsorted_rows_are_grouped_chronologically():
    data=frame(['2026-01-01T00:03:00Z','2026-01-01T00:00:00Z'],['later','earlier'])
    incidents=group_into_incidents(data,pd.Series([True,True],index=data.index))
    assert incidents[0]['start']==pd.Timestamp('2026-01-01T00:00:00Z')
    assert incidents[0]['duration_minutes']==4
