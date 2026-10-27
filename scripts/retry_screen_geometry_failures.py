"""Requeue geometry-label failures of an unfinished screening run, under its lock."""
import argparse
import fcntl
import json
from pathlib import Path
import tempfile


def requeue(root):
    root=Path(root)
    if not (root/'screening/config.json').is_file():
        raise ValueError('Not a screening output directory')
    with (root/'.screening.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Stop screening and wait for shutdown before retrying') from None
        if (root/'READY.json').exists():
            raise ValueError('Dataset already published; refusing to change its selected pages')
        paths=[]
        for split in ('train','val','test'):
            for path in (root/'screening'/split).glob('*.json'):
                record=json.loads(path.read_text())
                if (record.get('status')=='rejected' and
                    'label geometry mismatch' in record.get('error','')):
                    paths.append(path)
        backup=None
        if paths:
            backup=Path(tempfile.mkdtemp(prefix='geometry-rejections-',dir=root/'screening'))
            for path in paths:
                dest=backup/path.parent.name/path.name
                dest.parent.mkdir(exist_ok=True)
                path.rename(dest)
        return {'requeued':len(paths),'backup':str(backup) if backup else None}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(requeue(args.out)))
