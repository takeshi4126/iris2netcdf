# Copyright 2024 International Meteorological Consultant Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import math
import argparse
import os
import logging

import xarray as xr
import wradlib # Tested with wradlib version 2.0.3
import numpy as np
from cftime import date2num
from netCDF4 import Dataset #pylint: disable=no-name-in-module

# Earth Radius at the equator and at the pole in meter.
EQUATORIAL_RADIUS = 6378.1370 * 1000
POLAR_RADIUS = 6356.7523 * 1000
INSTITUTION = "Bangladesh Meteorological Department"
# INTERPOLATION = wradlib.ipol.Nearest
INTERPOLATION = wradlib.ipol.Idw

logger = logging.getLogger('iriwsraw2netcdf')
logger.setLevel(logging.DEBUG)

class RadarScan:
  """
  Class representing a radar scan data.
  """

  # Attributes given by the program arguments.
  image_height = None
  image_width = None
  file_name = None
  output_unit = None

  # Attributes read from the iris file.
  B = None
  beta = None
  generation_time = None
  product_name = None
  task_name = None
  dbz_polar = None
  azimuth = None
  elevation = None
  dtime = None
  rbins = None
  center = None
  radius = None # The radius of the observation range.
  bin_size = None # The size of the range bin in the radial direction.
  ground_height = None
  radar_height = None
  radar_site_name = None
  radar_number_bins = None

  # xarray object to store the converted data.
  data_array = None

  def __init__(self, output_unit, image_height, image_width):
    self.output_unit = output_unit
    self.image_height = image_height
    self.image_width = image_width

  def read_iris_file(self, iris_raw_data_file: str):
    """
    Read the iris raw data file and store some important parameters into this object.
    """
    with open(iris_raw_data_file, mode="rb") as f:
      iris_content = wradlib.io.read_iris(f, debug=False)
      self.__read_iris_header_part(iris_content)
      self.__read_iris_data_part(iris_content)
    self.file_name = os.path.basename(iris_raw_data_file)

  def __read_iris_header_part(self, iris_content: dict):
    """
    Read the header part of the iris raw data file.
    """
    product_hdr = iris_content['product_hdr']
    product_config = iris_content['product_hdr']['product_configuration']
    product_type = iris_content['product_type']
    if product_type != 'RAW':
      raise ValueError("This IRIS file is not a RAW data file. It is not supported.")

    self.beta = product_config['zr_exponent'] / 1000
    self.B = product_config['zr_constant'] / 1000

    # generation_time as a datetime object.
    self.generation_time = product_config['generation_time']
    # product name looks like 'PPI_A_Z   '. Watch out the trailing spaces.
    self.product_name = product_config['product_name']

    # task name looks like 'PPI_A     '. Watch out the trailing spaces.
    self.task_name = product_config['task_name']

    product_end = product_hdr['product_end']
    radar_lat = product_end['latitude']
    radar_lon = product_end['longitude']
    self.center = {'lat': radar_lat, 'lon': radar_lon}
    self.ground_height = product_end['ground_height']
    self.radar_height = product_end['radar_height']
    self.radar_site_name = product_end['site_name']
    self.radar_number_bins = product_end['number_bins']
    last_bin_range  = product_end['last_bin_range']
    self.radius = last_bin_range / 100 # It is stored in cm. Convert it to meter.

  def __read_iris_data_part(self, iris_content: dict):
    """
    Read the data part of the iris raw data file. We check only the first key of the data.
    """
    sweep_data = iris_content['data'][1]['sweep_data']

    # DB_DBZ is a ndarray of shape (360, 352). 360 degrees x 352 range bins.
    self.dbz_polar = sweep_data['DB_DBZ']

    # azimuth is of shape (360,) and contains the azimuth angles in degrees.
    self.azimuth = sweep_data['azimuth']

    # What's this elevation? It's 360 degrees but they vary between directions.
    self.elevation = sweep_data['elevation']

    # What is the dtime? The shape is (360, ) and it contains integers between 1 and 40.
    self.dtime = sweep_data['dtime']

    # Number of range bins for each azimuth. rbins is of shape (360, ). It contains 352 for all az.
    self.rbins = sweep_data['rbins']

    # Calculate the bin size
    self.bin_size = self.radius / self.rbins[0]

  def transform(self):
    """'
    The iris raw data file stores the radar scans in the polar coordinate system.
    This function transforms it into the cartesian coordinates. 
    Also, it converts the data from the radar reflectivity (dBZ) to rain intensity (mm/h) if specified so.
    """
    self.data_array = self.__polar_to_cartesian()
    if self.output_unit == 'mmh':
      # Convert the radar reflectivity from dbz to Z(mm6/m3)
      Z = self.data_array.wrl.trafo.idecibel()
      # Calculate the rain intensity from Z(mm6/m3) ot R(mm/h)
      R = Z.wrl.zr.z_to_r(a=self.B, b=self.beta)
      self.data_array.values = R
      logger.debug("min R = %f, max R = %f" % (R.min(), R.max()))
    elif self.output_unit == 'dbz':
      pass
    else:
      raise ValueError(f"Unsupported output_unit {self.output_unit}")
    # self.data_array.plot(x="x", y="y")

  def __polar_to_cartesian(self):
    """
    Convert the radar reflectivity (dbz) data in polar coordinates into the cartesian coordinates.
    Georef function of wradlib is used.
    """
    data = self.dbz_polar
    radar_location = (self.center['lon'], self.center['lat'], self.ground_height + self.radar_height)
    da = wradlib.georef.create_xarray_dataarray(
        data,
        r=np.arange(self.bin_size / 2, data.shape[1] * self.bin_size + self.bin_size / 2, self.bin_size),
        phi=self.azimuth,
        theta=self.elevation,
        site=radar_location,
        sweep_mode="azimuth_surveillance",
    )
    da.wrl.georef.georeference()
    # Reproject the data into EPSG:4326
    data_array = da.wrl.georef.reproject(trg_crs=wradlib.georef.epsg_to_osr(4326))
    return data_array

  def output_netcdf(self, out: str):
    """
    Output the transformed data into the netcdf4 format.
    The file contains 3 dimensions (time, lat, lon), however, the time will have only one element.
    The rain intensity data is saved as a grid data, like a forecasting model output, so that GIS a software can read it.
    """
    nc = Dataset(out, "w", format="NETCDF4")
    nc.title = self.file_name
    nc.history = "Created " + time.ctime(time.time())
    nc.Conventions = 'CF-1.6'
    nc.institution = INSTITUTION

    # Distance per 1-degree latitude
    lat_dist = (2 * math.pi * POLAR_RADIUS) / 360
    lon_dist = (2 * math.pi * EQUATORIAL_RADIUS) / 360

    center = self.center
    radius = self.radius

    # Create the grid data.
    lat_array = np.linspace(center['lat'] - radius / lat_dist, center['lat'] + radius / lat_dist, self.image_height)
    lon_array = np.linspace(center['lon'] - radius / lon_dist, center['lon'] + radius / lon_dist, self.image_width)
    cart = xr.Dataset(coords={"x": (["x"], lon_array), "y": (["y"], lat_array)})
    grid_data = self.data_array.wrl.comp.togrid(
      cart, radius=radius, center=(center['lon'], center['lat']), interpol=INTERPOLATION
    )
    # grid_data.plot()

    # Create dimensions
    nc.createDimension("time", 1)
    nc.createDimension("lat", len(lat_array))
    nc.createDimension("lon", len(lon_array))

    # Create variables (time, lat, lon, precipitation_rate)
    times = nc.createVariable("time","f4",("time",))
    times.units = "hours since 0001-01-01 00:00:00.0"
    times.calendar = "standard"
    dates = [self.generation_time]
    times[:] = date2num(dates,units=times.units,calendar=times.calendar)

    latitudes = nc.createVariable("lat","f4",("lat",))
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitude"
    latitudes[:] = lat_array

    longitudes = nc.createVariable("lon","f4",("lon",))
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitude"
    longitudes[:] = lon_array

    precipitation_rate = nc.createVariable("precipitation_rate","f4",("time", "lat", "lon"))
    precipitation_rate.units = "mm/h" if self.output_unit == "mmh" else "dbz"
    precipitation_rate.long_name = "precipitation rate"
    precipitation_rate[0,:,:] = grid_data.values

    logger.debug("min R = %f, max R = %f" % (precipitation_rate[0,:,:].min(), precipitation_rate[0,:,:].max()))

    nc.close()

def main():
  """
  The main function.
  """
  # Parse the arguments.
  parser = argparse.ArgumentParser(description='Convert IRIS raw data into netcdf format.')
  parser.add_argument('--input', type=str, required=True, help='Input file')
  parser.add_argument('--output', type=str, required=True, help='Output file')
  parser.add_argument('--output_unit', type=str, required=False, help='Output file', default='mmh', choices=['mmh', 'dbz'])
  parser.add_argument('--output_height', type=int, required=False, help='Output image height', default=360)
  parser.add_argument('--output_width', type=int, required=False, help='Output image width', default=360)

  args = parser.parse_args()

  input_file_path = args.input
  output_file_path = args.output

  # Instantiate the RadarScan and apply the conversion.
  scan = RadarScan(output_unit=args.output_unit, image_height=args.output_height, image_width=args.output_width)
  scan.read_iris_file(input_file_path)
  scan.transform()
  scan.output_netcdf(output_file_path)

if __name__ == '__main__':
  main()
