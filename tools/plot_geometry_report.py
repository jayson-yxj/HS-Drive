"""Static OpenCV report chart for reproducible M2 evidence."""
import argparse,json
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=json.load(open(a.report));img=np.full((620,1100,3),245,np.uint8)
    cv2.putText(img,'M2: adjacent-camera depth reprojection P95 (cm)',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(30,30,30),2)
    for i,(name,v) in enumerate(r['cross_view'].items()):
        y=70+i*38;value=v['p95_abs_m']*100
        cv2.putText(img,name,(20,y+18),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
        cv2.rectangle(img,(300,y),(300+int(min(value,5)*100),y+24),(120,160,40),-1)
        cv2.putText(img,f'{value:.3f} cm; n={v["count"]}',(820,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
    for i,line in enumerate(['Smooth visible surfaces only; 2-40m; discontinuities excluded.',
                              'Occluded samples are counted separately; no occupancy/free-space claim.',
                              f'Ego travel: {r["ego_displacement_m"]:.2f} m; motion gate: {r["motion_passed"]}']):
        cv2.putText(img,line,(20,550+i*23),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    if not cv2.imwrite(a.output,img):raise OSError(a.output)


if __name__=='__main__':main()
