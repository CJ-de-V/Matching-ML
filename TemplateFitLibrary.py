from iminuit import Minuit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iminuit import Minuit


def get_valid_values(df, feature):
    """
    Extract finite numerical values of a feature from a DataFrame.

    NaN and infinite values are excluded.
    """
    values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)

    return values[np.isfinite(values)]

def build_templates(
    mc_a,
    mc_b,
    feature,
    bins=50,
    value_range=None,
):
    """
    Construct normalised, binned templates for two MC classes.

    Parameters
    ----------
    mc_a, mc_b : pandas.DataFrame
        MC samples for classes A and B.

    feature : str
        Name of the feature to fit.

    bins : int or array-like
        Number of bins, or explicit bin edges.

    value_range : tuple or None
        Optional (minimum, maximum) range.
        If None, derive the range from both MC samples.

    Returns
    -------
    dict
        Shared bin edges, bin centres, and normalised templates.
    """
    a = get_valid_values(mc_a, feature)
    b = get_valid_values(mc_b, feature)

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both MC classes must contain valid feature values.")

    # Explicit bin edges take precedence over value_range.
    if np.ndim(bins) > 0 and not np.isscalar(bins):
        edges = np.asarray(bins, dtype=float)
    else:
        if value_range is None:
            value_range = (
                min(a.min(), b.min()),
                max(a.max(), b.max()),
            )

        edges = np.linspace(*value_range, int(bins) + 1)

    if (
        len(edges) < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Bin edges must be finite and strictly increasing.")

    # Both classes are histogrammed against precisely the same edges.
    counts_a, _ = np.histogram(a, bins=edges)
    counts_b, _ = np.histogram(b, bins=edges)

    # Normalise to unit probability. Events outside the selected range
    # are excluded, so these templates describe the in-range populations.
    if counts_a.sum() == 0 or counts_b.sum() == 0:
        raise ValueError("A template has no entries inside the selected range.")

    template_a = counts_a / counts_a.sum()
    template_b = counts_b / counts_b.sum()

    return {
        "feature": feature,
        "edges": edges,
        "centres": (edges[:-1] + edges[1:]) / 2,
        "counts_a": counts_a,
        "counts_b": counts_b,
        "template_a": template_a,
        "template_b": template_b,
    }

def fit_template_mixture(data, templates, feature=None, initial_fraction=0.5):
    """
    Fit a two-template mixture to a real-data distribution.

    The total expected yield is fixed to the observed number of entries
    within the histogram range. The class-A fraction is fitted.

    Parameters
    ----------
    data : pandas.DataFrame
        Real-data sample.

    templates : dict
        Output from build_templates().

    feature : str or None
        Feature name. Defaults to the feature used to build the templates.

    initial_fraction : float
        Initial guess for the class-A fraction.

    Returns
    -------
    dict
        Fit object, fitted fraction, uncertainties, histograms,
        and expected bin counts.
    """
    feature = feature or templates["feature"]

    if not 0 < initial_fraction < 1:
        raise ValueError("initial_fraction must lie strictly between 0 and 1.")

    edges = templates["edges"]
    template_a = templates["template_a"]
    template_b = templates["template_b"]

    values = get_valid_values(data, feature)

    # Use the same binning as the MC templates.
    data_counts, _ = np.histogram(values, bins=edges)
    n_total = int(data_counts.sum())

    if n_total == 0:
        raise ValueError("No valid data entries inside the fit range.")

    # Record how many entries are excluded from the histogram range.
    n_outside = len(values) - n_total

    def nll(f_A):
        """
        Negative log-likelihood, omitting terms constant in f_A.

        A zero expectation in a bin containing observed events gives
        an infinite negative log-likelihood, as it should.
        """
        probabilities = f_A * template_a + (1 - f_A) * template_b
        expected = n_total * probabilities

        occupied = data_counts > 0

        if np.any(expected[occupied] <= 0):
            return np.inf

        # The log term is evaluated only in occupied bins, avoiding log(0)
        # for empty bins. Empty bins still contribute their expected counts.
        return (
            expected.sum()
            - np.sum(data_counts[occupied] * np.log(expected[occupied]))
        )

    # iminuit minimises the negative log-likelihood.
    minimiser = Minuit(nll, f_A=initial_fraction)
    minimiser.errordef = Minuit.LIKELIHOOD
    minimiser.limits["f_A"] = (0, 1)

    minimiser.migrad()
    minimiser.hesse()

    if not minimiser.valid:
        raise RuntimeError(
            "Fit did not converge successfully. "
            "Inspect the templates, binning, and Minuit diagnostics."
        )

    f_A = minimiser.values["f_A"]
    f_A_err = minimiser.errors["f_A"]

    # Expected bin counts for each fitted component and their sum.
    expected_a = n_total * f_A * template_a
    expected_b = n_total * (1 - f_A) * template_b
    expected_total = expected_a + expected_b

    return {
        "feature": feature,
        "minuit": minimiser,
        "fraction_a": f_A,
        "fraction_a_error": f_A_err,
        "fraction_b": 1 - f_A,
        "fraction_b_error": f_A_err,
        "yield_a": n_total * f_A,
        "yield_b": n_total * (1 - f_A),
        "n_data": n_total,
        "n_outside": n_outside,
        "edges": edges,
        "centres": templates["centres"],
        "data_counts": data_counts,
        "expected_a": expected_a,
        "expected_b": expected_b,
        "expected_total": expected_total,
        "fit_valid": minimiser.valid,
    }

def plot_template_fit(result, templates):
    """
    Plot the observed data, fitted total, and fitted template components.

    Parameters
    ----------
    result : dict
        Output from fit_template_mixture().

    templates : dict
        Output from build_templates().
    """
    edges = result["edges"]
    centres = result["centres"]
    counts = result["data_counts"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Poisson counting uncertainties for the observed bin contents.
    ax.errorbar(
        centres,
        counts,
        yerr=np.sqrt(counts),
        fmt="o",
        color="black",
        markersize=4,
        label="Data",
    )

    # Overlay the expected counts from each fitted component.
    ax.stairs(
        result["expected_total"],
        edges,
        color="red",
        linewidth=2,
        label="Total fit",
    )

    ax.stairs(
        result["expected_a"],
        edges,
        linestyle="--",
        linewidth=1.5,
        label=f"Class A ({result['fraction_a']:.3f})",
    )

    ax.stairs(
        result["expected_b"],
        edges,
        linestyle="--",
        linewidth=1.5,
        label=f"Class B ({result['fraction_b']:.3f})",
    )

    # Display the fitted fractions and their uncertainties.
    fit_text = (
        f"$f_A$ = {result['fraction_a']:.4f}"
        f" $\\pm$ {result['fraction_a_error']:.4f}\n"
        f"$f_B$ = {result['fraction_b']:.4f}"
        f" $\\pm$ {result['fraction_b_error']:.4f}\n"
        f"$N_{{\\mathrm{{fit}}}}$ = {result['n_data']}\n"
        f"Excluded from range = {result['n_outside']}"
    )

    ax.text(
        0.97, 0.95, fit_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel(result["feature"])
    ax.set_ylabel("Entries / bin")
    ax.set_title(f"Two-template fit: {result['feature']}")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()