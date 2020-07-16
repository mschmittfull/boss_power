from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os

from nbodykit.lab import *
from nbodykit import setup_logging



def main():
    """
    Measure BOSS power spectrum.
    """

    # Parse command line arguments
    ap = ArgumentParser()

    ap.add_argument('--Nmesh', default=64, type=int,
        help='Nmesh used to compute power spectrum.')

    ap.add_argument('--download_dir', default='$SCRATCH/lss/sdss_data/', 
        help='Where to store/read downloaded data.')

    default_sample = 'DR12v5_CMASSLOWZTOT_South'
    #default_sample = 'DR12v5_LOWZ_South'
    ap.add_argument('--boss_sample', default=default_sample, 
        help='Which BOSS sample to use. See https://data.sdss.org/sas/dr12/boss/lss/')

    ap.add_argument('--out_dir', default='power/', 
        help='Folder where to store measured power spectrum.')

    ap.add_argument('--out_base', default='power', 
        help='Prefix for where to store measured power spectrum.')

    ap.add_argument('--plot', dest='plot', action='store_true', 
        help='Plot power spectra.')

    ap.add_argument('--subtract_shot', dest='subtract_shot', action='store_true',
        help='Subtract shot noise from monopole power spectrum.')

    cmd_args = ap.parse_args()


    # Setup things
    if cmd_args.plot:
        from nbodykit import style
        import matplotlib.pyplot as plt
        plt.style.use(style.notebook)

    setup_logging()

    # download the data to the current directory
    download_dir = os.path.expandvars(cmd_args.download_dir)
    print('download_dir:', download_dir)
    boss_sample = cmd_args.boss_sample
    download_data(download_dir, boss_sample=boss_sample)

    # NOTE: change this path if you downloaded the data somewhere else!
    data_path = os.path.join(download_dir, 'galaxy_%s.fits' % boss_sample)
    randoms_path = os.path.join(download_dir, 'random0_%s.fits' % boss_sample)

    # initialize the FITS catalog objects for data and randoms
    data = FITSCatalog(data_path)
    randoms = FITSCatalog(randoms_path)

    print('data columns = ', data.columns)
    print('randoms columns = ', randoms.columns)


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
    valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
    randoms = randoms[valid]

    # slice the data
    valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
    data = data[valid]

    Ngalaxies = data.csize
    print('Ngalaxies:', Ngalaxies)


    # the fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)


    randoms['WEIGHT'] = 1.0
    data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.0)


    # combine the data and randoms into a single catalog
    fkp = FKPCatalog(data, randoms)


    mesh = fkp.to_mesh(Nmesh=cmd_args.Nmesh, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

    # compute the multipoles
    r = ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.005, kmin=0.)

    for key in r.attrs:
        print("%s = %s" % (key, str(r.attrs[key])))

    # save results
    if not os.path.exists(cmd_args.out_dir):
        os.makedirs(cmd_args.out_dir)

    # save power to file, in nbodykit format
    fname = os.path.join(cmd_args.out_dir, 
        '%s_%s_Nmesh%d.nbk.dat' % (cmd_args.out_base, boss_sample, cmd_args.Nmesh))
    r.save(fname)
    print('Wrote %s' % fname)

    # Also save plain txt file
    poles = r.poles
    Nk = poles['k'].shape[0]
    mat = np.zeros((Nk, 4)) + np.nan
    header = 'boss_sample=%s\n' % boss_sample
    header += 'ZMIN=%g\n' % ZMIN
    header += 'ZMAX=%g\n' % ZMAX
    header += 'Nmesh=%d\n' % cmd_args.Nmesh
    header += 'Ngalaxies=%d\n' % Ngalaxies
    if cmd_args.subtract_shot:
        header += 'subtract_shot=True'
    else:
        header += 'subtract_shot=False'
    header += 'Columns: k, P_0, P_2, P_4'
    mat[:,0] = poles['k']
    if cmd_args.subtract_shot:
        mat[:,1] = poles['power_0'].real - r.attrs['shotnoise']
    else:
        mat[:,1] = poles['power_0'].real
    mat[:,2] = poles['power_2'].real
    mat[:,3] = poles['power_4'].real
    # save
    fname = os.path.join(cmd_args.out_dir, 
        '%s_%s_Nmesh%d.txt' % (cmd_args.out_base, boss_sample, cmd_args.Nmesh))
    np.savetxt(fname, mat, header=header)
    print('Wrote %s' % fname)

    if cmd_args.plot:
        # run code with --plot to plot
        for ell in [0, 2, 4]:
            label = r'$\ell=%d$' % (ell)
            P = poles['power_%d' %ell].real
            if cmd_args.subtract_shot:
                if ell == 0: P = P - r.attrs['shotnoise']
            plt.plot(poles['k'], poles['k']*P, label=label)

        # format the axes
        plt.legend(loc=0)
        plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
        plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \ \mathrm{Mpc}^2$]")
        plt.xlim(0.01, 0.25)

        fname = os.path.join(cmd_args.out_dir, 
            '%s_%s_Nmesh%d.pdf' % (cmd_args.out_base, boss_sample, cmd_args.Nmesh))
        plt.savefig(fname)
        print('Made %s' % fname)





def print_download_progress(count, block_size, total_size):
    import sys
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_data(download_dir, boss_sample='DR12v5_LOWZ_South'):
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
            'https://data.sdss.org/sas/dr12/boss/lss/random0_%s.fits.gz' % boss_sample]

    filenames = ['galaxy_%s.fits' % boss_sample, 
                 'random0_%s.fits' % boss_sample]

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
