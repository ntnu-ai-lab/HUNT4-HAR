from __future__ import print_function

import os
import csv
import sys
import json
import urllib
import shutil
import codecs
import argparse
import functools
import contextlib

try: # Try python3
  from urllib.request import urlopen 
  from urllib.parse import urlencode
  PYTHON_2 = False
  get_input = input 
except: # Fall back to python2
  from urllib2 import urlopen 
  from urllib import urlencode 
  PYTHON_2 = True
  get_input = raw_input



parser = argparse.ArgumentParser( description='API for accessing the HAR bulk inference server' )

parser.add_argument( 'method',
  nargs    = '?',
  help     = 'Name of api method to call'
)
parser.add_argument( '-o', '--output',
  default  = None,
  help     = 'Save returned data here if specified'
)
parser.add_argument( '--list-methods',
  action  = 'store_true',
  help    = 'List available API methods'
)
parser.add_argument( '--row-limit',
  default  = 20,
  type     = int,
  help     = 'Row limit when displaying csv. Use -1 to display everything'
)
parser.add_argument( '--docs',
  action = 'store_true',
  help   = 'Print docs for a given method'
)
parser.add_argument( '--host',
  default  = 'http://localhost',
  help     = 'Set which host to connect to'
)
parser.add_argument( '--port',
  default  = '19993',
  help     = 'Set which port to connect to'
)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#     Utils
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

API_METHODS = {}

def API_METHOD( url_suffix, desc=None, args={} ):
  '''
  Returns a decorator which formats the full url, and allows
  the decorated function to be called both from python and
  from the command line 
  '''
  def decorator( f ):
    # Decorated function, construct url from port and host and pass as input
    @functools.wraps( f )
    def decorated( host, port, *args, **kwargs ):
      url = '%s:%s'%( host, port ) + url_suffix
      return f( url, *args, **kwargs )

    # Register funciton in API_METHODS
    assert f.__name__ not in API_METHODS, 'Error: API Method with name "%s" is already registered' % f.__name__
    API_METHODS[ f.__name__.replace('_','-') ] = decorated

    # Add custom cli args that can be used for parsing later
    decorated.cli_args = ( desc, args )

    return decorated
  return decorator

def call( url, data=None ):

  # Add get arguemtns if provided
  if data:
    url = '%s?%s' % ( url, urlencode(data))
  try:
    # Send request to server
    resp = urlopen( url )
    # Make sure response is ok
    if not resp.code == 200:
      err =  'Received non-200 (%s) resp when calling "%s"\n%s'%( resp.code, url, resp.text )
      raise Exception( err )
  except Exception as e:
    err = 'An exception occured when calling "%s"\n%s'%( url, e )
    raise Exception( err )

  return resp 

def jsonify_args( **args ):
  '''Turn arguments into a dictionary of type key:json'''
  return { k : json.dumps(v) for k,v in args.items() }


def get_method_parser( api_method ):
  '''Generates an argument parser for an api method, using its decorator definition'''
  desc, args = api_method.cli_args
  parser = argparse.ArgumentParser( description=desc, add_help=False )

  for arg, params in sorted( args.items() ):
    parser.add_argument( '--%s'%arg.replace('_','-'), **params )

  return parser 


def list_methods():
  ''' Generate a string that lists available api methods '''
  s = '\nAvailable methods:\n  - '
  s += '\n  - '.join( sorted( API_METHODS))
  s += '\n\nExecute "<method_name> --docs" for more info about a specific method\n'
  return s

def get_docs( method_name ):
  ''' Generate a string that shows the documentation for a single api method '''
  f = API_METHODS[ method_name ]
  return '\n  DOCS: %s\n%s\n%s'%(method_name, '-'*80, get_method_parser( f ).format_help())

def save_to_disk( location, resp ):
  ''' Save the content of a file-like (response) object to disk '''
  with open( location, 'wb' ) as f:
    shutil.copyfileobj( resp, f )
  print( 'Successfully saved content to:', location )


def handle_json( resp ):
  print( json.dumps( json.load( resp ), indent=2 ))

def handle_csv( resp, limit ):

  rows = resp.readlines()
  if not rows:
    print( 'Empty CSV returned!' )
    return

  head = rows[:limit] if limit >= 0 else rows
  head = [ [ x.strip() for x in row.decode('ascii').split(',') ] for row in head ]
  width = [ min( 25, max( len(row[i]) for row in head )) for i in range(len(head[0])) ]
  for row in head:
    print( '  '.join( x.ljust(w) for x,w in zip( row, width )))

  remaining_rows = len( rows ) - len( head )
  if remaining_rows:
    print( '...' )
    print( '%s more rows truncated'%remaining_rows )
    print( 'Use "--row-limit -1" to display everything')


def handle_other( resp, content_type ):
  print( 'Received %s bytes of types %s from %s'%(resp.headers['content-length'], content_type, resp.url ))
  print( 'No handler is registered for handeling this type of output in the shell' )
  save = get_input( 'Save to disk? (y/N) ').lower()
  if save in ['y', 'yes']:
    location = get_input( 'Enter save location: ')
    save_to_disk( location, resp )
  else:
    print( 'Exiting' )

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#     API METHODS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@API_METHOD( '/api/subjects', 
  desc = 'Lists the names of subjects',
  args = dict(
    limit           = dict( type=int, default=10, help='Max rows to return' ),
    successful_only = dict( action='store_true', help='Whether to only return successfully processed subjects' )
))
def subject_names( url, limit, successful_only ):
  # Generate arguments
  params = jsonify_args(
    columns = ['name'],
    limit = limit,
    filters = dict( success=1 ) if successful_only else {}
  )
  # Call API
  return call( url, data=params )

@API_METHOD( '/api/subjects', 
  desc = 'List information about subjects',
  args = dict(
    limit           = dict( type=int, default=10, help='Max rows to return' ),
    successful_only = dict( action='store_true', help='Whether to only return successfully processed subjects' )
))
def subjects( url, limit, successful_only ):
  # Generate arguments
  params = jsonify_args(
    columns = ['name', 'start', 'end', 'success', 'path' ],
    limit = limit,
    filters = dict( success=1 ) if successful_only else {}
  )
  # Call API
  return call( url, data=params )


@API_METHOD( '/subject_file/%s/%s_timestamped_predictions.csv',
  desc = 'Download timestamped predictions csv for a single subject',
  args = dict(
    name = dict( required=True, help='Name of subject' ),
))
def timestamped_predictions( url, name ):
  # Build full url
  url = url % (name, name)
  # Call API
  return call( url )


@API_METHOD( '/subject_file/%s/%s_daily_overview.pdf',
  desc = 'Downloads a daily overview pdf plot for a single subject',
  args = dict(
    name = dict( required=True, help='Name of subject' ),
))
def daily_overview( url, name ):
  # Build full url
  url = url % (name,name)
  # Call API
  return call( url )



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#     MAIN FUNCTION
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def main( args ):

  if args.method:

    if not args.method in API_METHODS:
      print( 'Error: Unrecognized API method "%s".'%args.method )
      print( list_methods() )
      return 1

    if args.docs:
      print( get_docs( args.method ))
      return 0

    # Get method function
    method = API_METHODS[ args.method ]

    try:
      # Invoke API method, should return a file-like response object
      method_parser = get_method_parser( method )
      resp = method( args.host, args.port, **vars( method_parser.parse_known_args()[0] ))

      content_type = resp.info().gettype() if PYTHON_2 else resp.info().get_content_type()

      if args.output:
        save_to_disk( args.output, resp )
        return 0

      if content_type == 'application/json':
        handle_json( resp )

      elif content_type == 'text/csv':
        handle_csv( resp, limit=args.row_limit )

      else:
        handle_other( resp, content_type )

    except Exception as e:
      print( 'An error occured' )
      print( e )
      return 1

    finally:
      try: resp.close()
      except: pass


  elif args.list_methods:
    print( list_methods() )
    return 0

  else:
    print( 'No method specified!' )
    parser.print_help()
    return 1
    
  return 0



if __name__ == '__main__':
  args, _ = parser.parse_known_args()
  sys.exit( main( args ))