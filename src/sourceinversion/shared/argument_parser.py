import os

def add_mogi_parameters(parser):
    mogi = parser.add_argument_group('Mogi parameters')
    mogi.add_argument('--mogi-volume', type=float, nargs=2, default=[1e6, 2e7], help="Mogi volume range (default: %(default)s).")
    return parser


def add_penny_parameters(parser):
    penny = parser.add_argument_group('Penny parameters')
    penny.add_argument('--penny-radius', type=float, nargs=2, default=[800, 800], help="Penny radius range (default: %(default)s).")
    penny.add_argument('--penny-dp_mu', type=float, nargs=2, default=[0.0001, 0.01], help="Penny dp/mu range (default: %(default)s).")
    return parser


def add_spheroid_parameters(parser):
    spheroid = parser.add_argument_group('Spheroid parameters')
    spheroid.add_argument('--spheroid-strike', type=float, nargs=2, default=[0, 360], help="Spheroid strike range (default: %(default)s).")
    spheroid.add_argument('--spheroid-dip', type=float, nargs=2, default=[0, 90], help="Spheroid dip range (default: %(default)s).")
    spheroid.add_argument('--spheroid-ratio', type=float, nargs=2, default=[0.5, 1], help="Spheroid axis ratio range (default: %(default)s).")
    spheroid.add_argument('--spheroid-semi-axis', type=float, nargs=2, default=[500, 3000], help="Spheroid semi-axis range (default: %(default)s).")
    spheroid.add_argument('--spheroid-dp-mu', type=float, nargs=2, default=[0.0001, 0.01], help="Spheroid dp/mu range (default: %(default)s).")
    return parser


def add_okada_parameters(parser):
    # Okada / Dislocation (model id = 5 R)
    okada = parser.add_argument_group('Okada parameters')
    okada.add_argument('--okada-length', type=float, nargs=2, default=[1000, 5000], help="Fault length range (meters) (default: %(default)s).")
    okada.add_argument('--okada-width', type=float, nargs=2, default=[1000, 5000], help="Fault width range (meters) (default: %(default)s).")
    okada.add_argument('--okada-strike', type=float, nargs=2, default=[0, 360], help="Strike angle range (degrees) (default: %(default)s).")
    okada.add_argument('--okada-dip', type=float, nargs=2, default=[0, 90], help="Dip angle range (degrees) (default: %(default)s).")
    okada.add_argument('--okada-slip', type=float, nargs=2, default=[0, 10], help="Slip amount range (meters) (default: %(default)s).")
    okada.add_argument('--okada-rake', type=float, nargs=2, default=[0, 0], help="Rake angle range (degrees) (default: %(default)s).")
    okada.add_argument('--okada-opening', type=float, nargs=2, default=[0, 0], help="Opening displacement range (meters) (default: %(default)s).")
    return parser


def add_sampling_parameters(parser):
    sampling = parser.add_argument_group('Sampling parameters')
    sampling.add_argument('--sampling-id', type=str, choices=['0', '1'], default='0', help="Sampling ID, 0 for Natural Neighbor 1 for Bayesian (default: %(default)s).")
    sampling.add_argument('--p1', type=str, default='1000', help="Number of models to generate during the inversion (default: %(default)s).")
    sampling.add_argument('--p2', type=str, default='300', help="Number of models to keep after each iteration(default: %(default)s).")
    sampling.add_argument('--p3', type=str, default='30', help="Number of BI steps (default: %(default)s).")
    sampling.add_argument('--burn-in', type=str, default=2000, help="Number of initial models discarded as part of the burn-in phase (default: %(default)s).")
    return parser


def add_coordinates_parameters(parser):
    coords = parser.add_argument_group('Coordinates parameters')
    coords.add_argument('--x-range', type=float, nargs='*', default=[float('inf'), float('-inf')], help="X range (provide one or two values).")
    coords.add_argument('--y-range', type=float, nargs='*', default=[float('inf'), float('-inf')], help="Y range (provide one or two values).")
    coords.add_argument('--z-range', type=float, nargs=2, default=(0, 5000), help="Z range (default: %(default)s).")
    coords.add_argument('--scaling-box', type=float, default=0.15, help="Scaling factor for search box - x and y range - (default: %(default)s).")
    return parser