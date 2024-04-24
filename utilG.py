

import datetime
import time
from diskcache import Disk, FanoutCache
import numpy as np
from test.util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

unzipped_path = 'C:\\Users\\justm\\OneDrive\\Desktop\\New folder\\unzipped\\'

def getCacheHandle(scope_str):
    #Caches have an additional feature: memoizing decorator. The decorator wraps a callable and caches arguments and return values.
    return FanoutCache('disk_cache/' + scope_str,
                       disk=Disk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11)
    
def enumerateWithEstimate(
        iter,
        loop_title,
        start_ndx=0,
        print_ndx=4,
        jump=4,
        iter_len=None,
):
    """cool function to print out the progress of a loop"""
    if iter_len is None:
        iter_len = len(iter)

    if jump is None:
        jump = 2
        while jump ** 7 < iter_len:
            jump *= 2

    assert jump >= 2
    while print_ndx < start_ndx * jump:
        print_ndx *= jump

    log.warning("{} ----/{}, starting".format(
        loop_title,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            eta_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )
            # eta_sec = (now - start_time) / num_of_ndx_passed * num_of_ndx_remaining
            done_dt = datetime.datetime.fromtimestamp(start_ts + eta_sec)
            done_td = datetime.timedelta(seconds=eta_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                loop_title,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= jump 

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        loop_title,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

# irc = d-h-w, xyz = w-h-d
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_matrix):
    cri_a = coord_irc[::-1]
    # cri_a = np.array(coord_irc)[::-1]
    # origin_a = np.array(origin_xyz)
    # vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_matrix @ (cri_a * vxSize_xyz)) + origin_xyz
    return coords_xyz

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_matrix):
    # origin_a = np.array(origin_xyz)
    # vxSize_a = np.array(vxSize_xyz)
    # coord_a = np.array(coord_xyz)
    cri_a = ((coord_xyz - origin_xyz) @ np.linalg.inv(direction_matrix)) / vxSize_xyz
    return np.round(cri_a).astype(int)[::-1]
