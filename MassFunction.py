import numpy as np
import os
import collapsed_gibbs as DPGMM
import optparse as op
import configparser
import sys
import importlib.util


def is_opt_provided (parser, dest):
    for opt in parser._get_all_options():
        try:
            if opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]):
                return True
        except:
            if opt.dest == dest and opt._long_opts[0] in sys.argv[1:]:
                return True
    return False

def main():
    parser = op.OptionParser()
    
    parser.add_option("-i", "--input", type = "string", dest = "events_path", help = "Input folder")
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder")
    parser.add_option("--mmin", type = "float", dest = "mmin", help = "Minimum BH mass [Msun]", default = 3.)
    parser.add_option("--mmax", type = "float", dest = "mmax", help = "Maximum BH mass [Msun]", default = 200.)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected density")
    parser.add_option("--true_masses", type = "string", dest = "true_masses", help = "Simulated true masses")
    parser.add_option("--optfile", type = "string", dest = "optfile", help = "Options file. Passing command line options overrides optfile. It must contains ALL options")
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings", help = "Burnin, samples and step for MF sampling", default = '1000,100,10')
    parser.add_option("--samp_settings_ev", type = "string", dest = "samp_settings_ev", help = "Burnin, samples and step for single event sampling. If None, uses MF settings")
    parser.add_option("--mc_settings", type = "string", dest = "mc_settings", help = "Burnin and step for mass sampling", default = '100,10')
    parser.add_option("--hyperpars", type = "string", dest = "hyperpars", help = "MF hyperparameters (a0, b0, V0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 6 for reference", default = '3,10,4')
    parser.add_option("--hyperpars_ev", type = "string", dest = "hyperpars_ev", help = "Event hyperparameters (a0, b0, V0)", default = '3,3,4')
    parser.add_option("--alpha", type = "float", dest = "alpha0", help = "Internal (event) concentration parameter", default = 1.)
    parser.add_option("--gamma", type = "float", dest = "gamma0", help = "External (MF) concentration parameter", default = 1.)
    parser.add_option("-e", "--processed_events", dest = "process_events", action = 'store_false', default = True, help = "Disables event processing")
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 5.)
    parser.add_option("--nthreads", dest = "n_parallel_threads", type = "int", help = "Number of parallel threads to spawn", default = 8)
    parser.add_option("-v", "--verbose", dest = "verbose", action = 'store_true', default = False, help = "Display output")
    parser.add_option("-d", "--diagnostic", dest = "diagnostic", action = 'store_true', default = False, help = "Diagnostic plots")
    
    (options, args) = parser.parse_args()
    
    if options.optfile is not None:
        config = configparser.ConfigParser()
        config.read(options.optfile)
        opts = config['DEFAULT']
        for key, val in zip(vars(options).keys(), vars(options).values()):
            if not is_opt_provided(parser, key):
                vars(options)[key] = opts[key]
        if options.true_masses is 'None':
            options.true_masses = None
    options.hyperpars = [float(x) for x in options.hyperpars.split(',')]
    if options.hyperpars_ev is not None:
        options.hyperpars_ev = [float(x) for x in options.hyperpars_ev.split(',')]
    options.samp_settings = [int(x) for x in options.samp_settings.split(',')]
    if options.samp_settings_ev is not None:
        options.samp_settings_ev = [int(x) for x in options.samp_settings_ev.split(',')]
    options.mc_settings = [int(x) for x in options.mc_settings.split(',')]
    
    event_files = [options.events_path+f for f in os.listdir(options.events_path) if not f.startswith('.')]
    events      = []
    
    for event in event_files:
        events.append(np.genfromtxt(event))
    
    inj_density = None
    if options.inj_density_file is not None:
        inj_file_name = options.inj_density_file.split('/')[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(inj_file_name, options.inj_density_file)
        inj_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inj_module)
        inj_density = inj_module.injected_density
        
    
    sampler = DPGMM.CGSampler(events = events,
                              samp_settings = options.samp_settings,
                              samp_settings_ev = options.samp_settings_ev,
                              mass_chain_settings = options.mc_settings,
                              alpha0 = float(options.alpha0),
                              gamma0 = float(options.gamma0),
                              hyperpars_ev = options.hyperpars_ev,
                              hyperpars = options.hyperpars,
                              m_min = float(options.mmin),
                              m_max = float(options.mmax),
                              verbose = bool(options.verbose),
                              output_folder = options.output,
                              initial_cluster_number = int(options.initial_cluster_number),
                              process_events = bool(options.process_events),
                              n_parallel_threads = int(options.n_parallel_threads),
                              injected_density = inj_density,
                              true_masses = options.true_masses,
                              diagnostic = bool(options.diagnostic)
                              )
    sampler.run()
    
if __name__=='__main__':
    main()
