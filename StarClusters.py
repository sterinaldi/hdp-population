import numpy as np
from collapsed_gibbs import StarClusters
import optparse as op

def main():
    parser = op.OptionParser()

    parser.add_option("-c", "--catalog", type = "string", dest = "cat_file", help = "Catalog file", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same folder as catalog", default = None)
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings", help = "Burnin, samples and trimming for sampling", default = '10,100,1')
    parser.add_option("--hyperpars", type = "string", dest = "hyperpars", help = "Hyperparameters (L0, k0, nu0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 9 for reference", default = '1,5,5')
    parser.add_option("--alpha", type = "float", dest = "alpha", help = "DP concentration parameter", default = 1.)
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 50.)
    parser.add_option("--sigma_max", dest = "sigma_max", default = 20, help = "Max cluster variance")
    parser.add_option("--p_field", dest = "p_f", default = 0.5, help = "Probability threshold for field stars")
    parser.add_option("--dim", dest = "dim", default = None, help = "Dimensions (useful if the catalog contains additional columns, like magnitude or errors)")

    (options, args) = parser.parse_args()
    if options.cat_file is None:
        print('No catalog provided.')
        exit()
    if options.output is None:
        options.output = '/'.join(options.cat_file.split('/')[:-1])
    if options.dim is not None:
        options.dim = int(options.dim)
    options.hyperpars = [float(x) for x in options.hyperpars.split(',')]
    options.samp_settings = [int(x) for x in options.samp_settings.split(',')]
    catalog = np.genfromtxt(options.cat_file)
    
    sampler = StarClusters(catalog  = catalog,
                            burnin  = options.samp_settings[0],
                            n_draws = options.samp_settings[1],
                            step    = options.samp_settings[2],
                            alpha0  = float(options.alpha),
                            L       = options.hyperpars[0],
                            k       = options.hyperpars[1],
                            nu      = options.hyperpars[2],
                            output_folder = options.output,
                            initial_cluster_number = int(options.initial_cluster_number),
                            maximum_sigma_cluster  = float(options.sigma_max),
                            p_f_threshold = float(options.p_f),
                            dim = options.dim
                            )
    sampler.run()

if __name__ == '__main__':
    main()
