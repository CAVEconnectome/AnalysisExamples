# Parsing and uploading a bunch of neuroglancer states

This notebook is an example of a common workflow: Someone has a number of annotations in many neuroglancer states with tags indicating some classification per annotation and wants them uploaded to a CAVE table.
Here, the annotations are postsynaptic sites onto OPCs, and the annotations indicate if they are true synapses or not.
The notebook offers a good example of a parse-process-and-upload workflow, and shows a few common workarounds for data hygeine like inconsistent tag naming (in this case related to "true" vs "True").

