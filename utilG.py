

from diskcache import Disk, FanoutCache

unzipped_path = 'C:\\Users\\justm\\OneDrive\\Desktop\\New folder\\unzipped\\'

def getCacheHandle(scope_str):
    #Caches have an additional feature: memoizing decorator. The decorator wraps a callable and caches arguments and return values.
    return FanoutCache('disk_cache/' + scope_str,
                       disk=Disk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11)