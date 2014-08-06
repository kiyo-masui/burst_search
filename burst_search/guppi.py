"""Driver scripts and IO for Greenbank GUPPI data.

"""


import pyfits


def search_file(filename):
    """Simple dirver function to search a GUPPI file."""

    hdulist = pyfits.open(filename, 'readonly')
    
    parameters = parameters_from_header(hdulist)

    hdulist.close()


def parameters_from_header(hdulist):
    """Get data acqusition parameters for psrfits file header.

    Returns
    -------
    parameters : dict

    """

    parameters = {}

    print repr(hdulist[0].header)

    # XXX For now just fake it.
    parameters['cal_period'] = 64
    parameters['delta_t'] = 0.001
    parameters['nfreq'] = 4096
    parameters['freq0'] = 900e6
    parameters['delta_f'] = -200e6 / 4096

    return parameters


def read_records(hdulist, start_record=0, end_record=None):
    """Read and format records from GUPPI PSRFITS file."""




# This wants to be a class for sure.
def monitor_file(filename, time_block):
    """Monitor GUPPI file for new data and process in chunks."""
