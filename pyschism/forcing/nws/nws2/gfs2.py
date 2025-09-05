import os
from typing import Optional
from datetime import datetime, timedelta
import logging
import pathlib
import tempfile
from time import time
import glob
import multiprocessing as mp

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import xarray as xr
import pandas as pd

from pyschism.dates import nearest_cycle

logger = logging.getLogger(__name__)


class AWSGrib2Inventory:

    def __init__(
        self,
        start_date: Optional[datetime] = None,
        record: int = 1,
        pscr: Optional[str] = None,
        product: str = 'atmos',
        use_tempdir: bool = True,
        time_interval_hours: int = 1,
        forecast_mode: bool = False,
    ):
        """
        Download GFS data from AWS.
        This dataset GFS V16.3.0 starts on Feb 26, 2021.
        """
        self.start_date = nearest_cycle() if start_date is None else start_date
        if forecast_mode:
            self.start_date = self.start_date.replace(hour=3)
        self.pscr = pscr
        self.product = product
        self.use_tempdir = use_tempdir
        self.forecast_mode = forecast_mode
        # check start_date
        min_date = None
        response = self.s3.list_objects_v2(Bucket=self.bucket,
                                           Prefix="gfs.",
                                           Delimiter='/')
        if 'CommonPrefixes' in response:
            # Extract folder names, parse dates, and find the earliest date
            dates = []
            for obj in response['CommonPrefixes']:
                folder_name = obj['Prefix'].split('.')[1].rstrip(
                    '/')  # Extract the date part
                try:
                    date_obj = datetime.strptime(folder_name, '%Y%m%d')
                    dates.append(date_obj)
                except ValueError:
                    continue
            if dates:
                logger.info(
                    f"Available GFS data dates, from {min(dates)} to {max(dates)}"
                )
                min_date = min(dates)
        else:
            raise Exception("Unable to fetch GFS data dates from S3 bucket.")

        if min_date is not None and self.start_date < min_date:
            raise ValueError(
                f"Start date {self.start_date} is earlier than the earliest available date {min_date}"
            )

        self.forecast_cycle = self.start_date

        timevector = np.arange(
            self.start_date,
            self.start_date + timedelta(hours=record * 24 + 1),
            np.timedelta64(time_interval_hours, 'h')).astype(datetime)
        logger.info(f"{start_date} timevector is {timevector}")
        file_metadata = self.get_file_namelist(timevector, self.forecast_mode)
        for dt in timevector:
            if self.forecast_mode:
                outfile_name = f"gfs.{self.start_date.strftime('%Y%m%d')}/gfs.pgrb2.0p25.{self.start_date.strftime('%Y%m%d%H')}.{file_metadata[dt][0].split('.')[-1]}.grib2"
            else:
                outfile_name = f"gfs.{self.start_date.strftime('%Y%m%d')}/gfs.pgrb2.0p25.{dt.strftime('%Y%m%d%H')}.grib2"
            filename = pathlib.Path(self.tmpdir) / outfile_name
            filename.parent.mkdir(parents=True, exist_ok=True)
            if filename.exists():
                logger.info(f"file {filename} already exists, skipping")
                continue
            with open(filename, 'wb') as f:
                logger.debug(f"writing file {filename}")
                while (file_metadata.get(dt, None)):
                    try:
                        key = file_metadata[dt].pop(0)
                        logger.info(f"Downloading file {key} for {dt}")
                        self.s3.download_fileobj(self.bucket, key, f)
                        logger.info("Success!")
                        break
                    except:
                        if not file_metadata[dt]:
                            logger.info(f'No file for {dt}')
                            if os.path.exists(filename):
                                os.remove(filename)
                        else:
                            logger.info(
                                f'file {key} is not available, try next file')
                            continue

    def get_file_namelist(self, requested_dates, forecast_mode: bool = False):
        '''For a list of dates, prepare a list of GRIB files to download from AWS.

        In hindcast mode (forecast_mode = False), fetch the latest available data for each requested date.
          Iterates over initialization dates as well as cycles (00Z, 06Z, 12Z, 18Z) to extract only first portion of each model run.
            So each requested date will be extracted from the model run closest to the requested date.
       
        In forecast mode (forecast_mode = True), fetch the requested dates from the model run closest to the start date.
          Presume we are looking at only the most recent model run and must exctract all data from that one run.
          In this mode, the fxxx suffix of the GRIB filename is incremented.
          This suits operational forecast scenarios.      

        '''
        file_metadata = {}
        # hours ago from current time. used to figure out the number of forecast cycles.
        hours = (datetime.utcnow() - self.start_date).days * 24 + (
            datetime.utcnow() - self.start_date).seconds // 3600
        n_cycles = hours // 6 if hours < 25 else 4
        # requested_dates are the model forcing dates that we need to extract data for.
        for it, dt in enumerate(
                requested_dates[:n_cycles * 6 +
                                1]) if not forecast_mode else enumerate(
                                    requested_dates):
            i = 0
            if forecast_mode:
                levels = 1  # levels are the number of fxx time steps per requested date
                fhour = int(self.start_date.hour)
                # roll back to the previous day if the start date is at 00Z.
                date2 = (self.start_date - timedelta(days=1)).strftime(
                    '%Y%m%d'
                ) if self.start_date.hour == 0 else self.start_date.strftime(
                    '%Y%m%d')
            else:
                levels = 3
                fhour = int(dt.hour)
                date2 = (dt - timedelta(days=1)).strftime(
                    '%Y%m%d') if dt.hour == 0 else dt.strftime('%Y%m%d')
            # cycles are the forecast initilization hour: 00Z, 06Z, 12Z, 18Z.
            cycle_index = (fhour - 1) // 6
            while (levels):
                if forecast_mode:
                    # fhour2 must skip 000 because those steps do not contain all of the necessary data_vars.
                    cycle = self.fcst_cycles[cycle_index]
                    fhour2 = int(
                        (dt - self.start_date).total_seconds() // 3600) + 6
                    file_name = f"gfs.{date2}/{cycle}/{self.product}/gfs.t{cycle}z.pgrb2.0p25.f{fhour2:03d}"
                else:
                    cycle = self.fcst_cycles[cycle_index - i]
                    fhour2 = fhour + i * 6 if cycle_index == 0 else fhour - cycle_index * 6 + i * 6
                    file_name = f"gfs.{date2}/{cycle}/{self.product}/gfs.t{cycle}z.pgrb2.0p25.f{fhour2:03d}"
                logger.debug(f"cycle is {cycle}")
                file_metadata.setdefault(dt, []).append(file_name)
                levels -= 1
                i += 1

        if it < 25 and not forecast_mode:
            logger.info("fewer than 25 requested_dates were given.")
            date2 = (dt - timedelta(days=1)).strftime(
                '%Y%m%d') if dt.hour == 0 else dt.strftime('%Y%m%d')
            for it, dt in enumerate(requested_dates[n_cycles * 6 + 1:]):
                levels = 3
                i = 0
                hours = (dt - self.forecast_cycle).days * 24 + (
                    dt - self.forecast_cycle).seconds // 3600
                while (levels):
                    #starting from the last cycle
                    cycle = self.fcst_cycles[cycle_index - i]
                    fhour2 = (hours - (n_cycles - 1) * 6) + i * 6
                    file_metadata.setdefault(dt, []).append(
                        f"gfs.{date2}/{cycle}/{self.product}/gfs.t{cycle}z.pgrb2.0p25.f{fhour2:03d}"
                    )
                    levels -= 1
                    i += 1
        return file_metadata

    @property
    def bucket(self):
        return 'noaa-gfs-bdp-pds'

    @property
    def fcst_cycles(self):
        return ['00', '06', '12', '18']

    @property
    def s3(self):
        try:
            return self._s3
        except AttributeError:
            self._s3 = boto3.client('s3',
                                    config=Config(signature_version=UNSIGNED))
            return self._s3

    @property
    def tmpdir(self):
        if not hasattr(self, "_tmpdir"):
            if self.use_tempdir:
                self._tmpdir = tempfile.TemporaryDirectory(dir=self.pscr)
            else:
                logger.debug(f"self.pscr is {self.pscr}")
                self._tmpdir = pathlib.Path(self.pscr)
                logger.debug(f"self._tmpdir is {self._tmpdir}")
        return pathlib.Path(
            self._tmpdir.name) if self.use_tempdir else pathlib.Path(
                self._tmpdir)

    @property
    def files(self):
        grbfiles = glob.glob(
            f'{self.tmpdir}/gfs.{self.forecast_cycle.strftime("%Y%m%d")}/gfs.pgrb2.0p25.*.grib2'
        )
        grbfiles.sort()
        return grbfiles


def init_worker():
    """Initialize worker process with logging configuration."""
    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
        force=True)
    logging.getLogger("pyschism").setLevel(logging.DEBUG)


class GFS:

    def __init__(
        self,
        level=1,
        bbox=None,
        pscr=None,
    ):
        self.level = level
        self.bbox = bbox
        self.pscr = pscr
        self.record = 1

    def write(self,
              start_date,
              rnday,
              air: bool = True,
              prc: bool = True,
              rad: bool = True,
              use_tempdir: bool = True,
              forecast_mode: bool = False,
              time_interval_hours: int = 1):
        start_date = nearest_cycle() if start_date is None else start_date

        if ((start_date + timedelta(days=rnday))
                > datetime.utcnow()) | forecast_mode:
            logger.info(
                f'End date is beyond the current time, set rnday to 1 day and record to 5 days'
            )
            self.record = rnday  # number of days to record from a given start date
            rnday = 1

        end_date = start_date + timedelta(hours=rnday * self.record + 1)
        logger.info(f'start time is {start_date}, end time is {end_date}')

        if self.pscr is None:
            self.pscr = pathlib.Path(start_date.strftime("%Y%m%d"))
            self.pscr = pathlib.Path("sflux")
            self.pscr.mkdir(parents=True, exist_ok=True)

        datevector = np.arange(
            start_date,
            start_date + timedelta(days=rnday),
            np.timedelta64(1, 'D'),
            dtype='datetime64',
        )
        # why are we going after multiple dates-times?
        logger.info(f'datevector for GFS extraction is {datevector}')
        datevector = pd.to_datetime(datevector)
        npool = len(datevector) if len(
            datevector) < mp.cpu_count() / 2 else mp.cpu_count() / 2
        # TODO: Here we need multiprocessing to download all the GRIB files becasue that is the bottleneck when we are in forecasting mode.
        if len(datevector) > 1:
            logger.info(f'npool is {npool}')
            pool = mp.Pool(int(npool), initializer=init_worker)
            pool.starmap(self.gen_sflux,
                         [(istack + 1, date, air, prc, rad, use_tempdir,
                           time_interval_hours, forecast_mode)
                          for istack, date in enumerate(datevector)])
            pool.close()
        else:
            self.gen_sflux(1, datevector[0], air, prc, rad, use_tempdir,
                           time_interval_hours, forecast_mode)

    def gen_sflux(
        self,
        istack,
        date,
        air: bool = True,
        prc: bool = True,
        rad: bool = True,
        use_tempdir: bool = True,
        time_interval_hours: int = 1,
        forecast_mode: bool = False,
    ):
        inventory = AWSGrib2Inventory(date,
                                      self.record,
                                      self.pscr,
                                      use_tempdir=use_tempdir,
                                      time_interval_hours=time_interval_hours,
                                      forecast_mode=forecast_mode)
        grbfiles = inventory.files
        #cycle = date.hour

        prate = []
        dlwrf = []
        dswrf = []
        stmp = []
        spfh = []
        uwind = []
        vwind = []
        prmsl = []

        Vars = {
            'group1': {
                'sh2': ['174096', spfh],
                't2m': ['167', stmp],
                'u10': ['165', uwind],
                'v10': ['166', vwind]
            },
            'group2': {
                'prmsl': ['meanSea', prmsl]
            },
            'group3': {
                'prate': ['surface', prate]
            },
            'group4': {
                'dlwrf': ['surface', dlwrf],
                'dswrf': ['surface', dswrf]
            }
        }

        #Get lon/lat
        lon, lat, idx_ymin, idx_ymax, idx_xmin, idx_xmax = self.modified_latlon(
            grbfiles[0])

        times = []
        for ifile, file in enumerate(grbfiles):
            logger.info(f'file {ifile} is {file}')
            for key, value in Vars.items():
                if key == 'group1':
                    for key2, value2 in value.items():
                        ds = xr.open_dataset(
                            file,
                            engine='cfgrib',
                            backend_kwargs=dict(
                                filter_by_keys={'paramId': int(value2[0])}))
                        tmp = ds[key2][idx_ymin:idx_ymax + 1,
                                       idx_xmin:idx_xmax + 1].astype('float32')
                        value2[1].append(tmp[::-1, :])
                        ds.close()

                elif key == 'group2':
                    ds = xr.open_dataset(
                        file,
                        engine='cfgrib',
                        backend_kwargs=dict(
                            filter_by_keys={'typeOfLevel': 'meanSea'}))
                    for key2, value2 in value.items():
                        tmp = ds[key2][idx_ymin:idx_ymax + 1,
                                       idx_xmin:idx_xmax + 1].astype('float32')
                        value2[1].append(tmp[::-1, :])
                    times.append(ds.valid_time.values)
                    ds.close()

                elif key == 'group3':
                    ds = xr.open_dataset(
                        file,
                        engine='cfgrib',
                        backend_kwargs=dict(filter_by_keys={
                            'stepType': 'instant',
                            'typeOfLevel': 'surface'
                        }))
                    for key2, value2 in value.items():
                        tmp = ds[key2][idx_ymin:idx_ymax + 1,
                                       idx_xmin:idx_xmax + 1].astype('float32')
                        value2[1].append(tmp[::-1, :])
                    ds.close()

                else:
                    ds = xr.open_dataset(
                        file,
                        engine='cfgrib',
                        backend_kwargs=dict(filter_by_keys={
                            'stepType': 'avg',
                            'typeOfLevel': 'surface'
                        }))
                    for key2, value2 in value.items():
                        tmp = ds[key2][idx_ymin:idx_ymax + 1,
                                       idx_xmin:idx_xmax + 1].astype('float32')
                        value2[1].append(tmp[::-1, :])
                    ds.close()

        #write to netcdf
        bdate = date.strftime('%Y %m %d %H').split(' ')
        bdate = [int(q) for q in bdate[:4]] + [0]

        if air:
            ds = xr.Dataset(
                {
                    'stmp': (['time', 'ny_grid', 'nx_grid'], np.array(stmp)),
                    'spfh': (['time', 'ny_grid', 'nx_grid'], np.array(spfh)),
                    'uwind': (['time', 'ny_grid', 'nx_grid'], np.array(uwind)),
                    'vwind': (['time', 'ny_grid', 'nx_grid'], np.array(vwind)),
                    'prmsl': (['time', 'ny_grid', 'nx_grid'], np.array(prmsl)),
                },
                coords={
                    'time':
                    np.round((times - date.to_datetime64()) /
                             np.timedelta64(1, 'D'), 5).astype('float32'),
                    'lon': (['ny_grid', 'nx_grid'], lon),
                    'lat': (['ny_grid', 'nx_grid'], lat)
                })

            ds.time.attrs = {
                'long_name': 'Time',
                'standard_name': 'time',
                'base_date': bdate,
                'units': f"days since {date.strftime('%Y-%m-%d %H:00')} UTC",
            }

            ds.lat.attrs = {
                'units': 'degrees_north',
                'long_name': 'Latitude',
                'standard_name': 'latitude',
            }

            ds.lon.attrs = {
                'units': 'degrees_east',
                'long_name': 'Longitude',
                'standard_name': 'longitude',
            }

            ds.uwind.attrs = {
                'units': 'm/s',
                'long_name': '10m_above_ground/UGRD',
                'standard_name': 'eastward_wind'
            }

            ds.vwind.attrs = {
                'units': 'm/s',
                'long_name': '10m_above_ground/VGRD',
                'standard_name': 'northward_wind'
            }

            ds.spfh.attrs = {
                'units': 'kg kg-1',
                'long_name': '2m_above_ground/Specific Humidity',
                'standard_name': 'specific_humidity'
            }

            ds.prmsl.attrs = {
                'units': 'Pa',
                'long_name': 'Pressure reduced to MSL',
                'standard_name': 'air_pressure_at_sea_level'
            }

            ds.stmp.attrs = {
                'units': 'K',
                'long_name': '2m_above_ground/Temperature',
            }

            ds.to_netcdf(f'{self.pscr}/sflux_air_{self.level}.{istack:04d}.nc',
                         'w',
                         'NETCDF3_CLASSIC',
                         unlimited_dims='time')
            ds.close()

        if prc:
            ds = xr.Dataset(
                {
                    'prate': (['time', 'ny_grid', 'nx_grid'], np.array(prate)),
                },
                coords={
                    'time':
                    np.round((times - date.to_datetime64()) /
                             np.timedelta64(1, 'D'), 5).astype('float32'),
                    'lon': (['ny_grid', 'nx_grid'], lon),
                    'lat': (['ny_grid', 'nx_grid'], lat)
                })

            ds.time.attrs = {
                'long_name': 'Time',
                'standard_name': 'time',
                'base_date': bdate,
                'units': f"days since {date.strftime('%Y-%m-%d %H:00')} UTC"
            }

            ds.lat.attrs = {
                'units': 'degrees_north',
                'long_name': 'Latitude',
                'standard_name': 'latitude',
            }

            ds.lon.attrs = {
                'units': 'degrees_east',
                'long_name': 'Longitude',
                'standard_name': 'longitude',
            }

            ds.prate.attrs = {
                'units': 'kg m-2 s-1',
                'long_name': 'Precipitation rate'
            }

            ds.to_netcdf(f'{self.pscr}/sflux_prc_{self.level}.{istack:04d}.nc',
                         'w',
                         'NETCDF3_CLASSIC',
                         unlimited_dims='time')
            ds.close()

        if rad:
            ds = xr.Dataset(
                {
                    'dlwrf': (['time', 'ny_grid', 'nx_grid'], np.array(dlwrf)),
                    'dswrf': (['time', 'ny_grid', 'nx_grid'], np.array(dswrf)),
                },
                coords={
                    'time':
                    np.round((times - date.to_datetime64()) /
                             np.timedelta64(1, 'D'), 5).astype('float32'),
                    'lon': (['ny_grid', 'nx_grid'], lon),
                    'lat': (['ny_grid', 'nx_grid'], lat)
                })

            ds.time.attrs = {
                'long_name': 'Time',
                'standard_name': 'time',
                'base_date': bdate,
                'units': f"days since {date.strftime('%Y-%m-%d %H:00')} UTC"
            }

            ds.lat.attrs = {
                'units': 'degrees_north',
                'long_name': 'Latitude',
                'standard_name': 'latitude',
            }

            ds.lon.attrs = {
                'units': 'degrees_east',
                'long_name': 'Longitude',
                'standard_name': 'longitude',
            }

            ds.dlwrf.attrs = {
                'units': 'W m-2',
                'long_name': 'Downward short-wave radiation flux'
            }

            ds.dswrf.attrs = {
                'units': 'W m-2',
                'long_name': 'Downward long-wave radiation flux'
            }

            ds.to_netcdf(f'{self.pscr}/sflux_rad_{self.level}.{istack:04d}.nc',
                         'w',
                         'NETCDF3_CLASSIC',
                         unlimited_dims='time')
            ds.close()

    def modified_latlon(self, grbfile):
        xmin, xmax, ymin, ymax = self.bbox.xmin, self.bbox.xmax, self.bbox.ymin, self.bbox.ymax
        xmin = xmin + 360 if xmin < 0 else xmin
        xmax = xmax + 360 if xmax < 0 else xmax
        ds = xr.open_dataset(grbfile,
                             engine='cfgrib',
                             backend_kwargs=dict(filter_by_keys={
                                 'stepType': 'instant',
                                 'typeOfLevel': 'surface'
                             }))
        lon = ds.longitude.astype('float32')
        lat = ds.latitude.astype('float32')
        lon_idxs = np.where((lon.values >= xmin - 1.0)
                            & (lon.values <= xmax + 1.0))[0]
        lat_idxs = np.where((lat.values >= ymin - 1.0)
                            & (lat.values <= ymax + 1.0))[0]
        idx_ymin = lat_idxs[0]
        idx_ymax = lat_idxs[-1]
        idx_xmin = lon_idxs[0]
        idx_xmax = lon_idxs[-1]
        lon2 = lon[lon_idxs]
        lat2 = lat[lat_idxs]
        idxs = np.where(lon2 > 180)
        lon2[idxs] -= 360
        logger.info(
            f'idx_ymin is {idx_ymin}, idx_ymax is {idx_ymax}, idx_xmin is {idx_xmin}, idx_xmax is {idx_xmax}'
        )
        #make sure lat is in ascending order
        nx_grid, ny_grid = np.meshgrid(lon2, lat2[::-1])

        ds.close()

        return nx_grid, ny_grid, idx_ymin, idx_ymax, idx_xmin, idx_xmax
