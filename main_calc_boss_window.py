from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os

#from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit import CurrentMPIComm
from nbodykit.algorithms import SurveyDataPairCount
from nbodykit.source.catalog.file import FITSCatalog
from nbodykit.cosmology import cosmology
from nbodykit.source.catalog.uniform import MPIRandomState


def main():
    """
    Calculate BOSS window function multipoles.
    
    See https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.html#nbodykit.algorithms.SurveyDataPairCount

    and 

    https://arxiv.org/pdf/1607.03150.pdf eq 22
    """

    # Parse command line arguments
    ap = ArgumentParser()

    ap.add_argument('--rmin', default=100.0, type=float,
        help='Number of bins for separation r between pairs.')

    ap.add_argument('--rmax', default=500.0, type=float,
        help='Number of bins for separation r between pairs.')

    ap.add_argument('--Nr', default=20, type=int,
        help='Number of bins for separation r between pairs.')

    ap.add_argument('--Nmu', default=10, type=int,
        help='Number of bins for angle mu w.r.t. line of sight.')

    ap.add_argument('--download_dir', default='$SCRATCH/lss/sdss_data/', 
        help='Where to store/read downloaded data.')

    default_sample = 'DR12v5_CMASSLOWZTOT_South'
    #default_sample = 'DR12v5_LOWZ_South'
    ap.add_argument('--boss_sample', default=default_sample, 
        help='Which BOSS sample to use. See https://data.sdss.org/sas/dr12/boss/lss/')

    ap.add_argument('--out_dir', default='window/', 
        help='Folder where to store measured window function.')

    ap.add_argument('--out_base', default='paircount', 
        help='Prefix for where to store measured window function.')

    ap.add_argument('--FKP', default=0, type=int,
        help='Include FKP weight.')

    ap.add_argument('--randoms1_catalog_id', default=0, type=int,
        help='ID for randoms1 catalog')

    ap.add_argument('--randoms2_catalog_id', default=1, type=int,
        help='ID for randoms2 catalog')

    ap.add_argument('--subsample_fraction', default=1e-4, type=float,
        help='If less than 1, use random subsample of randoms.')

    cmd_args = ap.parse_args()

    setup_logging()
    comm = CurrentMPIComm.get()

    # download the data to the current directory
    download_dir = os.path.expandvars(cmd_args.download_dir)
    if comm.rank == 0:
        print('download_dir:', download_dir)
    boss_sample = cmd_args.boss_sample

    if comm.rank == 0:
        download_data(download_dir, boss_sample=boss_sample,
            random_catalog_id=cmd_args.randoms1_catalog_id)
        download_data(download_dir, boss_sample=boss_sample,
            random_catalog_id=cmd_args.randoms2_catalog_id)

    # NOTE: change this path if you downloaded the data somewhere else!
    randoms1_path = os.path.join(download_dir, 'random%d_%s.fits' % (
        cmd_args.randoms1_catalog_id, boss_sample))
    randoms2_path = os.path.join(download_dir, 'random%d_%s.fits' % (
        cmd_args.randoms2_catalog_id, boss_sample))

    # initialize the FITS catalog objects for data and randoms
    randoms1 = FITSCatalog(randoms1_path)
    randoms2 = FITSCatalog(randoms2_path)

    if comm.rank == 0:
        print('randoms1 columns = ', randoms1.columns)
        print('randoms2 columns = ', randoms2.columns)


    # Select redshift range
    if boss_sample in ['DR12v5_LOWZ_South', 'DR12v5_LOWZ_North']:
        ZMIN = 0.15
        ZMAX = 0.43
    elif boss_sample in ['DR12v5_CMASS_South', 'DR12v5_CMASS_North']:
        ZMIN = 0.43
        ZMAX = 0.7
    elif boss_sample in ['DR12v5_CMASSLOWZTOT_South', 'DR12v5_CMASSLOWZTOT_North']:
        ZMIN = 0.5
        ZMAX = 0.75
    else:
        raise Exception('Must specify ZMIN and ZMAX for boss_sample=%s' % str(boss_sample))

    # slice the randoms
    valid1 = (randoms1['Z'] > ZMIN)&(randoms1['Z'] < ZMAX)
    randoms1 = randoms1[valid1]
    valid2 = (randoms2['Z'] > ZMIN)&(randoms2['Z'] < ZMAX)
    randoms2 = randoms2[valid2]

    if cmd_args.subsample_fraction < 1.0:
        # Create random subsamples
        rng1 = MPIRandomState(randoms1.comm, seed=123, size=randoms1.size)
        rr1 = rng1.uniform(0.0, 1.0, itemshape=(1,))
        randoms1 = randoms1[rr1[:,0] < cmd_args.subsample_fraction]
        rng2 = MPIRandomState(randoms2.comm, seed=456, size=randoms2.size)
        rr2 = rng2.uniform(0.0, 1.0, itemshape=(1,))
        randoms2 = randoms2[rr2[:,0] < cmd_args.subsample_fraction]

    Nrandoms1 = randoms1.csize
    Nrandoms2 = randoms2.csize
    if comm.rank == 0:
        print('Nrandoms1:', Nrandoms1)
        print('Nrandoms2:', Nrandoms2)


    # weights
    if cmd_args.FKP == 0:
        randoms1['Weight'] = 1.0
        randoms2['Weight'] = 1.0
    else:
        randoms1['Weight'] = randoms1['WEIGHT_FKP']
        randoms2['Weight'] = randoms2['WEIGHT_FKP']



    # the fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

    # bins for separation
    edges_r = np.logspace(np.log10(cmd_args.rmin), np.log10(cmd_args.rmax),
        num=cmd_args.Nr+1)
    print('edges_r', edges_r)

    if comm.rank == 0:
        print("Start pair count...")
    paircount = SurveyDataPairCount(mode='2d', first=randoms1, second=randoms2,
     edges=edges_r,
     cosmo=cosmo, 
     Nmu=cmd_args.Nmu, 
     pimax=None, ra='RA', dec='DEC', redshift='Z', weight='Weight', 
     show_progress=True, domain_factor=4)

    if comm.rank == 0:
        print("Done pair count")

    if comm.rank == 0:
        print('paircount', paircount)
        for key in paircount.attrs:
            print("%s = %s" % (key, str(paircount.attrs[key])))



    # save results
    if comm.rank == 0:
        if not os.path.exists(cmd_args.out_dir):
            os.makedirs(cmd_args.out_dir)

    # save window to file, in nbodykit format
    out_file_base = os.path.join(cmd_args.out_dir, 
        '%s_%s_rmin%.1f_rmax%.1f_Nr%d_Nmu%d_randID1%d_randID2%d_SUB%g_FKP%d' % (
            cmd_args.out_base, boss_sample, 
            cmd_args.rmin, cmd_args.rmax, cmd_args.Nr, cmd_args.Nmu,
            cmd_args.randoms1_catalog_id, cmd_args.randoms2_catalog_id,
            cmd_args.subsample_fraction, cmd_args.FKP
            ))

    fname = '%s.nbk.dat' % out_file_base
    paircount.save(fname)
    print('Wrote %s' % fname)



def print_download_progress(count, block_size, total_size):
    import sys
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_data(download_dir, boss_sample='DR12v5_LOWZ_South',
    random_catalog_id=0):
    """
    Download the FITS data needed for this notebook to the specified directory.

    Parameters
    ----------
    download_dir : str
        the data will be downloaded to this directory
    """
    from six.moves import urllib
    import shutil
    import gzip

    urls = ['https://data.sdss.org/sas/dr12/boss/lss/galaxy_%s.fits.gz' % boss_sample,
            'https://data.sdss.org/sas/dr12/boss/lss/random%d_%s.fits.gz' % (
                random_catalog_id, boss_sample)]

    filenames = ['galaxy_%s.fits' % boss_sample, 
                 'random%d_%s.fits' % (random_catalog_id, boss_sample)]

    # download both files
    for i, url in enumerate(urls):

        # the download path
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)
        final_path = os.path.join(download_dir, filenames[i])

        # do not re-download
        if not os.path.exists(final_path):
            print("Downloading %s" % url)

            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # Download the file from the internet.
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path,
                                                      reporthook=print_download_progress)

            print()
            print("Download finished. Extracting files.")

            # unzip the file
            with gzip.open(file_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print("Done.")
        else:
            print("Data has already been downloaded.")


if __name__ == '__main__':
    main()
