# --- trackmate_bridge.py ------------------------------------------------------
import imagej
import scyjava as sj
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.util import map_array


# One JVM per interpreter – initialise lazily so importing the module is cheap
_ij = None


def _get_ij():
    global _ij
    if _ij is not None:
        return _ij
    # Use the full Fiji distribution so TrackMate classes are on the classpath
    _ij = imagej.init("sc.fiji:fiji", mode="headless", add_legacy=True)

    return _ij


def _np_to_img(stack_np: np.ndarray):
    """
    Convert a (T,Z,Y,X) or (T,Y,X) uint16 np stack to an ImageJ HyperStack.
    Z slices are optional – TrackMate ignores them if Z=1.
    """
    ij = _get_ij()
    # Ensure 4D (T,Z,Y,X); TrackMate expects at least XYT
    if stack_np.ndim == 3:  # (T,Y,X)
        stack_np = stack_np[:, None, ...]
    T, Z, Y, X = stack_np.shape
    img = ij.py.to_img(stack_np)  # wraps without copy
    # Set metadata so TrackMate sees time/Z
    imp = ij.py.to_imageplus(img)
    imp.setDimensions(1, Z, T)  # channels=1
    imp.setOpenAsHyperStack(True)
    return imp


def run_trackmate_from_cc(
    labels: np.ndarray,
    *,
    link_max_dist: float = 10.0,
    gap_closing_max_dist: float = 10.0,
    gap_closing_max_frame_gap: int = 2,
):
    """
    Headless TrackMate run that **skips detection**: spots are the centroids
    of connected components in each binary frame.

    Returns the usual DataFrame with
        ['t', 'x', 'y', 'z', 'track_id', 'spot_id'].
    """
    ij = _get_ij()

    # ---------------------------------------------------------------------
    # Build a dummy ImagePlus (TrackMate demands one for metadata only)
    t, y, x = labels.shape
    blank = np.zeros_like(labels, dtype=np.uint16)
    imp = ij.py.to_imageplus(ij.py.to_img(blank))
    imp.setDimensions(1, 1, t)
    imp.setOpenAsHyperStack(True)
    # ---------------------------------------------------------------------

    # Java plumbing
    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    Spot = sj.jimport("fiji.plugin.trackmate.Spot")
    SpotCollection = sj.jimport("fiji.plugin.trackmate.SpotCollection")
    LAPTrackerFactory = sj.jimport("fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory")
    ManualDetectorFactory = sj.jimport("fiji.plugin.trackmate.detection.ManualDetectorFactory")
    TrackMate = sj.jimport("fiji.plugin.trackmate.TrackMate")

    model = Model()
    spots = SpotCollection()
    quality = 1.0
    Integer = sj.jimport("java.lang.Integer")
    # ---------------------------------------------------------------------
    # Centroids → Spot objects
    for frame, img in enumerate(labels):
        for prop in regionprops(img):
            y_c, x_c = prop.centroid  # skimage returns row,col
            radius_px = prop.equivalent_diameter / 2.0
            s = Spot(float(x_c), float(y_c), 0.0, radius_px, quality)
            s.putFeature("FRAME", float(frame))
            s.putFeature("LABEL", float(prop.label))
            spots.add(s, Integer(frame))
    model.setSpots(spots, False)
    # ---------------------------------------------------------------------

    settings = Settings(imp)
    settings.detectorFactory = ManualDetectorFactory()
    settings.detectorSettings = {}  # empty map is fine
    settings.trackerFactory = LAPTrackerFactory()
    tracker_settings = settings.trackerFactory.getDefaultSettings()  # <- never None

    # ---- patch the handful of parameters we actually vary ------------------
    Double = sj.jimport("java.lang.Double")
    Integer = sj.jimport("java.lang.Integer")
    Boolean = sj.jimport("java.lang.Boolean")

    tracker_settings.put("LINKING_MAX_DISTANCE", Double(link_max_dist))
    tracker_settings.put("GAP_CLOSING_MAX_DISTANCE", Double(gap_closing_max_dist))
    tracker_settings.put("MAX_FRAME_GAP", Integer(gap_closing_max_frame_gap))
    tracker_settings.put("ALLOW_TRACK_SPLITTING", Boolean(False))
    tracker_settings.put("ALLOW_TRACK_MERGING", Boolean(False))

    settings.trackerSettings = tracker_settings
    
    tm = TrackMate(model, settings)
    if not tm.checkInput() or not tm.process():
        raise RuntimeError(tm.getErrorMessage())

    # ------------- back to pandas ---------------------------------------
    records = []
    for spot in model.getSpots().iterable(False):
        frame = spot.getFeature("FRAME")
        x = spot.getFeature("POSITION_X")
        y = spot.getFeature("POSITION_Y")
        z = spot.getFeature("POSITION_Z")
        spot_id = spot.ID()
        track_id = model.getTrackModel().trackIDOf(spot)
        # convert nan to -1 
        if track_id is None:
            track_id = -1 # this is the default value for unassigned spots
        else:
            track_id = int(track_id) + 1 # TrackMate IDs are 0-based so we convert it
        label_value = spot.getFeature("LABEL")
        records.append((frame, x, y, z, track_id, spot_id, label_value))

    cols = ["t", "x", "y", "z", "track_id", "spot_id", 'label']
    return pd.DataFrame(records, columns=cols).sort_values(["track_id", "t"])


def remap_segmentation(df: pd.DataFrame,
                       segmentation: list,
                       timepoint_column: str = 'timepoint',
                       label_column: str = 'label',
                       measure_column: str = 'ERK') -> np.ndarray:
    # Turn your df into a sorted numpy array [time, label, measure]
    tracked = (df[[timepoint_column, label_column, measure_column]]
               .sort_values(timepoint_column)
               .to_numpy())
    
    # Find the unique timepoints (in ascending order) and group the rows by them
    unique_tps = np.unique(tracked[:, 0]).astype(int)
    groups = [tracked[tracked[:, 0] == tp] for tp in unique_tps]
    
    n_seg = len(segmentation)
    max_tp = unique_tps.max()
    
    # Decide whether segmentation is absolute-indexed (len >= max_tp+1)
    # or relative (one entry for each unique timepoint)
    if n_seg >= max_tp + 1:
        # absolute: segmentation[tp]
        get_img = lambda tp, idx: segmentation[tp]
    elif n_seg == len(unique_tps):
        # relative: segmentation[idx]
        get_img = lambda tp, idx: segmentation[idx]
    else:
        raise ValueError(
            f"Can't align your segmentation list (len={n_seg}) with "
            f"{len(unique_tps)} tracked timepoints "
            f"(max timepoint={max_tp})."
        )
    
    # Now remap each frame
    remapped = []
    for idx, (tp, grp) in enumerate(zip(unique_tps, groups)):
        img = get_img(tp, idx)
        labels = grp[:, 1].astype(int)
        measures = grp[:, 2]
        remapped.append(map_array(img, labels, measures))
    
    return np.stack(remapped, axis=0)



