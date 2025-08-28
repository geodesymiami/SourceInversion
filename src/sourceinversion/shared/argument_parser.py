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
    spheroid.add_argument('--spheroid-axis-ratio', type=float, nargs=2, default=[0.5, 1], help="Spheroid axis ratio range (default: %(default)s).")
    spheroid.add_argument('--spheroid-semi-axis', type=float, nargs=2, default=[500, 3000], help="Spheroid semi-axis range (default: %(default)s).")
    spheroid.add_argument('--spheroid-dp_mu', type=float, nargs=2, default=[0.0001, 0.01], help="Spheroid dp/mu range (default: %(default)s).")
    return parser


def add_okada_parameters(parser):
    # Okada / Dislocation (model id = 5 R)
    okada = parser.add_argument_group('Okada parameters')
    parser.add_argument('--okada-length', type=float, nargs=2, default=[1000, 5000], help="Fault length range (meters) (default: %(default)s).")
    parser.add_argument('--okada-width', type=float, nargs=2, default=[1000, 5000], help="Fault width range (meters) (default: %(default)s).")
    parser.add_argument('--okada-strike', type=float, nargs=2, default=[0, 360], help="Strike angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-dip', type=float, nargs=2, default=[0, 90], help="Dip angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-slip', type=float, nargs=2, default=[0, 10], help="Slip amount range (meters) (default: %(default)s).")
    parser.add_argument('--okada-rake', type=float, nargs=2, default=[0, 0], help="Rake angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-opening', type=float, nargs=2, default=[0.0, 1.0], help="Opening displacement range (meters) (default: %(default)s).")
    return parser