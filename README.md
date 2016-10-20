# pydrs

This is a very simple attempt to make a data reduction pipeline for VLT/SPHERE observations.

The esorex software must be installed, and the SPHERE recipes should be 0.15.0 (otherwise some intermediate products are not saved). 

The pipeline is imported as:
> from pydrs import DRS

And is called as:
> data_red = DRS('Starname', 'path to data')

The pipeline should ask you some questions along the way, to reduce the dark, flat, centering frames, etc. You can create blacklists for each steps to ignore some of the files you don't want.

