import numpy as np


def construct_lambda_schedule(num_windows):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    manually optimized by YTZ
    """

    A = int(0.35 * num_windows)
    B = int(0.30 * num_windows)
    C = num_windows - A - B

    # Empirically, we see the largest variance in std <du/dl> near the endpoints in the nonbonded
    # terms. Bonded terms are roughly linear. So we add more lambda windows at the endpoint to
    # help improve convergence.
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.25, A, endpoint=False),
            np.linspace(0.25, 0.75, B, endpoint=False),
            np.linspace(0.75, 1.0, C, endpoint=True),
        ]
    )

    assert len(lambda_schedule) == num_windows

    return lambda_schedule


def validate_lambda_schedule(lambda_schedule, num_windows):
    """Must go monotonically from 0 to 1 in num_windows steps"""
    assert lambda_schedule[0] == 0.0
    assert lambda_schedule[-1] == 1.0
    assert len(lambda_schedule) == num_windows
    assert ((lambda_schedule[1:] - lambda_schedule[:-1]) > 0).all()


def interpolate_pre_optimized_protocol(pre_optimized_protocol, num_windows):
    xp = np.linspace(0, 1, len(pre_optimized_protocol))
    x_interp = np.linspace(0, 1, num_windows)
    lambda_schedule = np.interp(x_interp, xp, pre_optimized_protocol)

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_conversion_lambda_schedule(num_windows):
    lambda_schedule = np.linspace(0, 1, num_windows)
    validate_lambda_schedule(lambda_schedule, num_windows)
    return lambda_schedule


def construct_absolute_lambda_schedule_complex(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.50 * num_windows)
    C = num_windows - A - B

    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.1, A, endpoint=False),
            np.linspace(0.1, 0.3, B, endpoint=False),
            np.linspace(0.3, 1.0, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_absolute_lambda_schedule_solvent(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.66 * num_windows)
    D = 1  # need only one window from 0.6 to 1.0
    C = num_windows - A - B - D

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 0.50, C, endpoint=True),
            [1.0],
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_pre_optimized_absolute_lambda_schedule_solvent(num_windows, nonbonded_cutoff=1.2):
    """Linearly interpolate a lambda schedule pre-optimized for solvent decoupling

    Notes
    -----
    * Generated by post-processing ~half a dozen solvent decoupling calculations
        (see context in description of PR #538)
    * Assumes nonbonded cutoff = 1.2 nm
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    # fmt: off
    solvent_decoupling_protocol = np.array(
        [0., 0.02154097, 0.0305478, 0.03747918, 0.0432925, 0.04841349, 0.05303288, 0.05729336, 0.06128111, 0.0650162,
         0.06854392, 0.07186945, 0.07505386, 0.07809426, 0.08097656, 0.08378378, 0.08652228, 0.08910844, 0.09170097,
         0.09415532, 0.0965975, 0.09894146, 0.10125901, 0.10349315, 0.1057036, 0.10782406, 0.10995297, 0.11196338,
         0.11404105, 0.11597311, 0.11799029, 0.11989214, 0.12179616, 0.12367442, 0.12544245, 0.12730977, 0.12904358,
         0.13080329, 0.13255268, 0.13418286, 0.13594787, 0.13760607, 0.13920917, 0.14090233, 0.14247115, 0.14403571,
         0.14563762, 0.14712597, 0.14863463, 0.1501709, 0.1516045, 0.15306237, 0.15457974, 0.15599668, 0.15739867,
         0.1588833, 0.1602667, 0.16158698, 0.16306219, 0.16443643, 0.16571203, 0.1671053, 0.16844875, 0.16969885,
         0.17095515, 0.17229892, 0.17355947, 0.17474395, 0.17606238, 0.17735235, 0.1785562, 0.1797194, 0.18102615,
         0.18224503, 0.18338315, 0.18454735, 0.18579297, 0.18695968, 0.18805265, 0.18920557, 0.1904094, 0.1915372,
         0.1925929, 0.19370481, 0.19486737, 0.19595772, 0.19698288, 0.19803636, 0.1991899, 0.20028, 0.20131035,
         0.20232168, 0.20348772, 0.20458663, 0.2056212, 0.20659485, 0.20774405, 0.20884764, 0.20989276, 0.2108857,
         0.2120116, 0.21316817, 0.21427184, 0.21532528, 0.21650709, 0.21773745, 0.21890783, 0.22002229, 0.22133134,
         0.2226356, 0.22387771, 0.22515419, 0.22662608, 0.22803088, 0.22940172, 0.23108277, 0.2327005, 0.23438922,
         0.23634133, 0.23822652, 0.2405842, 0.24292293, 0.24588996, 0.24922462, 0.25322387, 0.25836924, 0.26533154,
         0.27964026, 0.29688698, 0.31934273, 0.34495637, 0.37706286, 0.4246625, 0.5712542, 1.]
    )
    # fmt: on

    lambda_schedule = interpolate_pre_optimized_protocol(solvent_decoupling_protocol, num_windows)

    return lambda_schedule

def construct_relative_lambda_schedule(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0
    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded cutoff = 1.2 nm
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.15 * num_windows)
    B = int(0.60 * num_windows)
    C = num_windows - A - B

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.00, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 1.00, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule