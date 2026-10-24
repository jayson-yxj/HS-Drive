"""One static-only, exact-identity map region cache. Shapely2 immutable values."""
from shapely.ops import unary_union


class StaticRegionCache:
    def __init__(self):
        self.regions={};self.hits=0;self.misses=0

    def region(self,map_sha,polygons,allowed,inset):
        lanes=tuple(sorted(allowed))
        if not lanes or any(k not in polygons for k in lanes):raise ValueError('missing_lane_geometry')
        # No rounded geometry or hidden map identity, WKB includes holes/components.
        key=(map_sha,lanes,tuple(polygons[k].wkb for k in lanes),float(inset))
        if key not in self.regions:
            self.misses+=1
            self.regions[key]=unary_union([polygons[k] for k in lanes]).buffer(-inset)
        else:self.hits+=1
        return self.regions[key]
