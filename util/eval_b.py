import sys
import os

# Add project root to path so we can import local util.trackeval
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from util import trackeval

eval_config = {
    "USE_PARALLEL": True,
    "NUM_PARALLEL_CORES": 8,
    "BREAK_ON_ERROR": True,
    "RETURN_ON_ERROR": False,
    "LOG_ON_ERROR": "",
    "PRINT_RESULTS": False,
    "PRINT_ONLY_COMBINED": False,
    "PRINT_CONFIG": False,
    "TIME_PROGRESS": False,
    "DISPLAY_LESS_PROGRESS": False,
    "OUTPUT_SUMMARY": True,
    "OUTPUT_EMPTY_CLASSES": False,
    "OUTPUT_DETAILED": True,
    "PLOT_CURVES": True,
}

dataset_config = {
    "PRINT_CONFIG": False,
    "GT_FOLDER": "",
    "TRACKERS_FOLDER": "",
    "OUTPUT_FOLDER": "",
    "TRACKERS_TO_EVAL": [""],
    "CLASSES_TO_EVAL": ["pedestrian"],
    "BENCHMARK": "MOT17",
    "SPLIT_TO_EVAL": "",
    "INPUT_AS_ZIP": False,
    "DO_PREPROC": False,
    "TRACKER_SUB_FOLDER": "",
    "OUTPUT_SUB_FOLDER": "",
    "TRACKER_DISPLAY_NAMES": None,
    "SEQMAP_FOLDER": None,
    "SEQMAP_FILE": "",
    "SEQ_INFO": None,
    "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
    "SKIP_SPLIT_FOL": True,
}

metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}


def track_eval(tracker_folder, gt_folder, output_folder, seqmap):
    # Init metrics
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    # Init evaluator
    evaluator = trackeval.Evaluator(eval_config)

    # Init dataset
    dataset_config["TRACKERS_FOLDER"] = tracker_folder
    dataset_config["GT_FOLDER"] = gt_folder
    dataset_config["OUTPUT_FOLDER"] = output_folder
    dataset_config["SEQMAP_FILE"] = seqmap
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    # Run evaluation
    res, success = evaluator.evaluate(dataset_list, metrics_list)

    idf1 = round(res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["Identity"]["IDF1"] * 100, 3)
    hota = round(np.mean(res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["HOTA"]) * 100, 3)
    assa = round(np.mean(res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["AssA"]) * 100, 3)
    deta = round(np.mean(res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["DetA"]) * 100, 3)
    mota = round(res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["CLEAR"]["MOTA"] * 100, 3)
    idsw = res["MotChallenge2DBox"][""]["COMBINED_SEQ"]["pedestrian"]["CLEAR"]["IDSW"]

    return {"IDF1": float(idf1), "HOTA": float(hota), "MOTA": float(mota), "ASSA": float(assa), "DETA": float(deta), "IDSW": int(idsw)}


if __name__ == '__main__':
    tracker_folder = "/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot_Henning/logs/b/tracker"
    gt_folder = "/images/SegmentationDistillation/data/BFT/test"
    output_folder = "/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot_Henning/out"
    seqmap = "/images/SegmentationDistillation/data/BFT/test_seqmap.txt"
    b = track_eval(tracker_folder, gt_folder, output_folder, seqmap)

    print("IDF1: {:.3f}, HOTA: {:.3f}, MOTA: {:.3f}, ASSA: {:.3f}, DETA: {:.3f}, IDSW: {}".format(b["IDF1"], b["HOTA"], b["MOTA"], b["ASSA"], b["DETA"], b["IDSW"]))
