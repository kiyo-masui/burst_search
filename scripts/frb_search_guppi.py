#! /usr/bin/python
"""Program to monitor directory for new data and search for fast radio bursts
in GUPPI data.

Processes all data found in the search directory

"""

import time
import argparse
import glob
import multiprocessing
from os import path

import yaml
import watchdog.observers
import watchdog.events

# Command line arguments.
parser = argparse.ArgumentParser(description='Search GUPPI data for FRBs.')
parser.add_argument("parameter_file", metavar="parameter file", type=str,
                    help="YAML file with setup parameters.")
parser.add_argument(
    "--search-dir",
    nargs=1,
    help=("Directory containing data to be searched or for"
          " new data. Overrides the value provied in parameter file."),
    )
parser.add_argument(
        "--no-real-time",
        help="Do not do real-time search, just process pre-existing files.",
        action='store_true',
        )
parser.add_argument(
        "--no-pre-existing",
        help="Do search pre-existing files, just perform real-time search.",
        action='store_true',
        )


# Function that processes a file.
def process_file(filename, bunch, of, parameters, return_queue):
    pass


# What to do with new files.
class NewFileHandler(watchdog.events.PatternMatchingEventHandler):
    
    def set_up(self, bandpass_filename, time_block, remove_noise_cal,
               output_directory, process_queue, return_queue):
        pass

    def on_created(self, event):
        print "Here"
        print event
        if event.is_directory:
            return
        filename = event.src_path
        # process_file(...)



if __name__ == "__main__":
    args = parser.parse_args()

    # Read parameter file.
    with open(args.parameter_file) as f:
        parameters = yaml.safe_load(f.read())
    if args.search_dir:
        parameters["search_directory"] = args.search_dir
    parameters["search_directory"] = path.expanduser(parameters["search_directory"])
    file_pattern = path.join(parameters['search_directory'],
                             parameters['filename_match_pattern'])
    print file_pattern

    if not args.no_real_time:
        process_queue = multiprocessing.Queue()
        return_queue = multiprocessing.Queue()
        event_handler = NewFileHandler(patterns=[file_pattern],
                                       case_sensitive=True)
        # Set up and start new file monitor.
        observer = watchdog.observers.Observer()
        observer.schedule(event_handler, parameters['search_directory'],
                          recursive=False)
        observer.start()

    try:
        if not args.no_pre_existing:
            # First process all files that already exist in search directory.
            files_to_search = glob.glob(file_pattern)

        # Now enter holding loop until keyboard interupt.
        while not args.no_real_time:
            time.sleep(1)
    except KeyboardInterrupt:
        if not args.no_real_time:
            observer.stop()
            observer.join()

