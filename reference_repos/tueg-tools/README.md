# tueg-tools
Tools for working with the Temple University Hospital EEG (TUHEEG) Corpus

## Download EEG data
The Temple group provide 
[various scripts/instructions for downloading their dataset](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml), but in
 our experience some users struggle to make these work, and there are known limitations. Hence 
  tueg-tools includes a (hopefully) easy-to-use, platform-independent method for downloading the
   full dataset (TUEG), a specific subset (defined by root URL), or some size-limited portion of the
    specified subset for anyone with basic python and git skills.
  
Just clone this repo, cd to its folder and then in python...

```python
import tueg_tools
ds = tueg_tools.Dataset('C:/Your_path_to_where_data_should_be_downloaded')
ds.download('https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/',
             username='Your Username', password='Your Password', maxSize=10**11)
```

Note that in the above code, the optional argument ```maxSize=10**11``` limits the download to the
 first 100 GB (10^11 bytes). 
 
You will need to insert your own username and password.  You can obtain these
 by requesting access from the maintainers of the TUHEEG Corpus, as described on [their downloads
  page](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml).
 
 The URL specified in this example points to the TUAB subset. You can specify the URL for
  any of the TUHEEG subsets* and this function will walk through their sitemap and reproduce the
   folder structure on your machine (stopping early if maxSize is specified).
   
*Hopefully. Only TUAB and TUEG have been tested.

## Iterating over the dataset

The tueg_tools.Dataset class includes various generator methods to simplify iterating over
 multiple sessions/recordings in the dataset. Look through the code to get a full idea of what's
  been implemented so far. As a basic example:
  
```python
ds = tueg_tools.Dataset('C:/Your_Path_to_where_data_was_downloaded')
for eeg in ds.eeg_gen():
    # Do something with each of the EEG records in this dataset.
```
 
 
   