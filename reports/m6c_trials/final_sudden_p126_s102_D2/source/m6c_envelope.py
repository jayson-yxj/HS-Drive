"""Empirical stopping bounds: research support, never a universal physics proof."""
import bisect,math
from dataclasses import dataclass


@dataclass
class Estimate:
    distance_m: float | None
    supported: bool
    reason: str | None
    duration_s: float | None


class StoppingEnvelope:
    def __init__(self,model):self.model=model

    def estimate(self,speed,grade,action='brake_001',variant='C'):
        if not math.isfinite(speed) or not math.isfinite(grade) or speed<0:return Estimate(None,False,'InvalidState',None)
        if speed<=.02:return Estimate(0.,True,None,0.)
        if abs(grade)>.03:return Estimate(None,False,'GradeOutOfSupport',None)
        if action=='coast' and grade<-.005:return Estimate(None,False,'DownhillCoastDidNotStop',None)
        cells=self.model['tables'][action];edges=self.model['speed_edges'];index=bisect.bisect_left(edges,speed)-1
        if index<0:index=0
        if index>=len(cells) or cells[index] is None:return Estimate(None,False,'SpeedOutOfSupport',None)
        cell=cells[index];distance=cell['max_distance_m'];duration=cell['max_duration_s']
        if variant=='C':
            delay=self.model['response_delay_s'];extra_grade=max(0,cell['min_grade']-grade+self.model['grade_uncertainty'])
            distance+=speed*delay+.5*9.81*max(0,-grade)*delay**2+.5*9.81*extra_grade*duration**2+self.model['unmodeled_distance_m']
        return Estimate(float(distance),True,None,float(duration))
