# Assignment 1: Heartbeat Detection

Heartbeat (QRS complex) detection in ECG records. We implement and evaluate a real-time detection algorithm following *H.C. Chen and S.W. Chen. 2003. A moving average based filtering system with its application to real-time qrs detection*.

![Algorithm pipeline](/figs/algo.png)

![Peak detections on the s20011 LTST record](/figs/detects.png)

![Peak detections on the 121 MIT record with challenging baseline wander](/figs/wander.png)


## Usage

Experimental code with all figures is available as a jupyter notebook in `as1.ipynb`.

Or run the detector as a commandline tool. The detected peaks will be saved as an annotation file with the same name.
```
./detect.py [path_to_wfdb_record]
```

See `report.pdf` for implementation and evaluation details.
