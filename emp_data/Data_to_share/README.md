The dataset was kindly shared by Gozzi's Functional Neuroimaging lab (Pagani et al., 2021, Nature Communications)

For each of the 20 mice, we extracted BOLD timeseries (1870 timepoints) and saved them as .mat files, with each column representing one ROI and each row one timepoint.
Regions/voxels are ordered per hemisphere. 
This means that the first 86 columns represent all the ROIs (from Primary Motor Area to Paraflocculus) from one hemisphere, and that starting from 87 they represent the other hemisphere with the exact same ordering.
The ordering is the same that can be found in the macro_atlas_n_172_rois_excel_final.xlsx

DATA SPECIFICS:

- 7T MRI scanner (72-mm birdcage transmit coil and a 4-channel solenoid coil for signal reception)
- Total scanning time: ~30 minutes (1869 s)
- Repetition time [TR]: 1000 ms (also used as the dt of BOLD simulation)
- Echo time [TE]: 15 ms
- Matrix: 192 x 192
- Field of view: 2 x 2 cm
- Coronal slices: 24
- Slice thickness: 500 µm
