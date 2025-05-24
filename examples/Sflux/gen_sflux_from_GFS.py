#python routine to pull gfs data and create netcdf files based on startdate 
#and rnday (# of days). Data written to gfs_{date}.nc files
from datetime import datetime

from pyschism.mesh.hgrid import Hgrid
from pyschism.forcing.nws.nws2.gfs2 import GFS

if __name__ == '__main__':
    startdate = datetime(2025, 4, 22)
    rnday = 4
    record = 1
    hgrid = Hgrid.open('./hgrid.gr3', crs='epsg:4326')
    pscr = '.'
    gfs = GFS(start_date=startdate, rnday=rnday, pscr=pscr, record=record, bbox=hgrid.bbox)
