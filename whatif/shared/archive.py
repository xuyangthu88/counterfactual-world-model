"""Consolidate run shards without altering the simulator or any Stage 07 data."""
import csv
from .actions import OUT


def pack(path, groups):
    with path.open('w',newline='') as dst:
        writer=None
        for kind,files in groups.items():
            for f in files:
                with f.open() as source:
                    for row in csv.DictReader(source):
                        row={'split':kind,**row}
                        if writer is None:
                            writer=csv.DictWriter(dst,fieldnames=row,lineterminator='\n');writer.writeheader()
                        writer.writerow(row)


if __name__=='__main__':
    def files(prefix,steps):
        return [p for p in sorted(OUT.glob(prefix+'-*.csv')) if p.name.endswith('-steps.csv')==steps]
    pack(OUT/'validation.csv',{'source':files('val',False),'transfer':files('confirm',False)})
    pack(OUT/'control.csv',{'test':files('test',False)})
    pack(OUT/'trajectory.csv',{'test':files('test',True)})
