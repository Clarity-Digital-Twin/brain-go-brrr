""" Library of tools to help with handling files in the Temple University EEG corpus.
    These tools are focussed on navigating the database, rather than specific tasks
    related to text processing or EEG analysis.

    Example usage:
    import tueg_tools
    ds = tueg_tools.Dataset('/myPathToDataset')
    for eeg in ds.eeg_gen():
        # Do something with each of the EEG records in this dataset.

    Created by David Western, July 2020.
"""

import datetime
import os
from pathlib import Path
import re

#TODO: Various instances of 'raise NotImplementedError' in this file indicate unimplemented features.

class Dataset:
    """An object representing the full Temple dataset (or perhaps a subset of the corpus,
    depending on input params), as stored at a particular location.
    """
    def __init__(self, path):
        self.path = path

    def session_gen(self, mode='sequential'):
        """Generator to iterate over eeg recording sessions in the dataset."""

        if mode is 'sequential':
            pass
        else:
            raise NotImplementedError

        # Identify sessions under this path, based on TUEG naming convention.
        for file in os.scandir(self.path):
            if re.match('v\d+.\d+.\d+',file.name):
                edf_path = os.path.join(file.path,'edf')
                if os.path.isdir(os.path.join(edf_path,'eval')):
                    # TUAB dataset, which has two extra folder levels for eval/train and normal/abnormal
                    for subdir in os.scandir(edf_path):
                        for subdir2 in os.scandir(subdir):
                            for tcp_dir in os.scandir(subdir2):
                                for arb_dir in os.scandir(tcp_dir.path):
                                    for subj_dir in os.scandir(arb_dir.path):
                                        for ses_dir in os.scandir(subj_dir.path):
                                            yield Session(ses_dir.path)
                else:
                    # Full TUEG dataset:
                    for tcp_dir in os.scandir(edf_path):
                        for arb_dir in os.scandir(tcp_dir.path):
                            for subj_dir in os.scandir(arb_dir.path):
                                for ses_dir in os.scandir(subj_dir.path):
                                    yield Session(ses_dir.path)
            elif file.name in ['normal', 'abnormal']:
                # TUAB2 dataset (file structure used in Schirrmeister et al's auto-eeg-diagnosis-example)
                edf_path = os.path.join(file.path, 'edf')
                for subdir in os.scandir(edf_path):
                    if subdir.name in ['eval', 'train']:
                        for tcp_dir in os.scandir(subdir):
                            for arb_dir in os.scandir(tcp_dir.path):
                                for subj_dir in os.scandir(arb_dir.path):
                                    for ses_dir in os.scandir(subj_dir.path):
                                        yield Session(ses_dir.path)

    def eeg_gen(self, mode='sequential'):
        """Generator to iterate over eeg records in the dataset."""

        if mode is not 'sequential':
            raise NotImplementedError

        sessions = self.session_gen(mode=mode)
        for ses in sessions:
            eegs = ses.eeg_gen()
            for eeg in eegs:
                yield eeg

    def report_gen(self):
        """Generator to iterate over eeg reports (text) in the dataset."""
        raise NotImplementedError

    def download(self, url, username=None, password=None, maxSize=float('Inf'),
                 currPath=None):
        """Download all data from the specified URL. The function will walk recursively through the
        directory, downloading all children. The resulting folder structure will mimic the directory
        structure at the host.

        Keyword arguments:
        url -- the web address for the top directory of the data to be downloaded.
        username -- credentials for data access (default = 'nedc_tuh_eeg').
        password -- credentials for data access. To request access, ask someone who knows more than
            you, or just sign up at https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php.
        maxSize -- the function will stop after downloading this many bytes. e.g. for 100 GB use
            maxTotalSize=10**11 (default=Infinity; stopping when all data is downloaded).
        currPath -- Parent directory in which to save files. If no value is specified (currPath=None),
            then the self.path (i.e. the path specified on instantiating this dataset object) will be used.

        Example usage:
        import tueg_tools
        ds = tueg_tools.Dataset('C:/YourPath')
        # N.B. Below, maxSize=10*11 limits the download to the first 100 GB.
        ds.download('https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/',
            username='Your Username', password='Your Password', maxSize=10**11)
        """

        from bs4 import BeautifulSoup
        import requests

        if password==None:
            raise ValueError('You must provide a password.')
        if currPath==None:
            currPath = self.path

        ignoreTheseAttrs = ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory',
                            'www.isip.piconepress.com']

        page = requests.get(url, auth=(username,password)).text
        soup = BeautifulSoup(page, 'html.parser')
        nodes = [n for n in soup.find_all('a') if not n.text in ignoreTheseAttrs]
        subDirs = [n.get('href') for n in nodes if n.get('href').endswith('/')]
        fileNames = [n.get('href') for n in nodes if not n.get('href').endswith('/')]
        for fileName in fileNames:
            ff = os.path.join(currPath, fileName)
            if os.path.isfile(ff):
                print("Already got "+ff+". Skipping...")
                continue
            fURL = url+fileName
            r = requests.head(fURL, auth=(username,password), allow_redirects=True)
            size = int(r.headers['content-length'])
            if size>maxSize:
                print("Bytes limit would be breached. Stopping before downloading {}-bytes file {}.".format(size,fURL))
                return False,maxSize
            else:
                if not maxSize==float('inf'):
                    print("{} bytes to go. ".format(maxSize), end="")
                print("Downloading "+fURL+".")
                r = requests.get(fURL, auth=(username,password), allow_redirects=True, stream=True)
                size = len(r.content)
                open(ff,'wb').write(r.content)
                maxSize-=size
        for sd in subDirs:
            newPath = os.path.join(currPath,sd)
            if not os.path.isdir(newPath):
                os.mkdir(newPath)
            cont,maxSize = self.download(url+sd, username=username, password=password, maxSize=maxSize,
                          currPath=newPath)
            if not cont:
                return False,maxSize

        return True,maxSize


        # parse_re = re.compile('href="([^"]*)".*(..-...-.... ..:..).*?(\d+[^\s<]*|-)')
        #
        # nGot = 0
        # try:
        #     html = urllib.urlopen(url).read()
        # except IOError as e:
        #     print()
        #     'error fetching %s: %s' % (url, e)
        #     return
        # if not url.endswith('/'):
        #     url += '/'
        # files = parse_re.findall(html)
        # dirs = []
        # print()
        # url + ' :'
        # print()
        # '%4d file' % len(files) + 's' * (len(files) != 1)
        # for name, date, size in files:
        #     if size.strip() == '-':
        #         size = 'dir'
        #     if name.endswith('/'):
        #         dirs += [name]
        #     print()
        #     '%5s  %s  %s' % (size, date, name)
        #
        # for dir in dirs:
        #     print()
        #     list_apache_dir(url + dir)




class Subject:
    """Each instance of this class represents a subject in the dataset.
    """
    def __init__(self, path):
        self.path = path
        raise NotImplementedError


    def session_gen(self):
        """Generator to iterate over all eeg records for this subject."""
        raise NotImplementedError

    def eeg_gen(self):
        """Generator to iterate over all eeg records for this subject."""
        raise NotImplementedError

    def report_gen(self):
        """Generator to iterate over all eeg reports (text) for this subject."""
        raise NotImplementedError


class Session:
    """Each instance of this class represents an EEG recording session in the dataset.
    """
    def __init__(self,path=None):
        p = Path(path)
        if len(p.parts[-1].split('_')) != 4:
            print(path)
            raise ValueError("Path does not end with a valid session ID. "+p.parts[-1])
        self.path = path
        self.subjectID = p.parts[-2]
        self.ses_no,year,month,day = p.parts[-1].split('_')
        self.date = datetime.date(int(year),int(month),int(day))

    def eeg_gen(self, mode='sequential'):
        """Generator to iterate over all eeg records in this session."""
        if mode == 'sequential':
            for file in os.scandir(self.path):
                if file.name.endswith(".edf"):
                    yield EEG_Record(path=file.path)
        else:
            raise NotImplementedError

    def report_gen(self):
        for file in os.scandir(self.path):
            if file.name.endswith(".txt"):
                yield EEG_Report(path=file.path)



class EEG_Record:
    """Each instance of this class represents an EEG recording session in the dataset.
    """
    def __init__(self, path):
        self.path = path
        p = Path(path)
        if len(p.parts) >= 2:
            ses_no,year,month,day = p.parts[-2].split('_')
            self.date = datetime.date(int(year),int(month),int(day))
        else:
            self.date = None
        metadata = p.parts[-1].split('_')
        if len(metadata) == 3:
            self.subjectID = metadata[0]
            self.ses_no = metadata[1]
            self.token = metadata[2].split('.')[0]
        else:
            self.subjectID = self.ses_no = self.token = None


class EEG_Report:
    """Each instance of this class represents an EEG recording session in the dataset.
    """
    def __init__(self, path=None):
        self.path = path

    def get_age_sex(self):
        import eegreportparser as erp
        ageStr,sex = erp.GetDemographics(self.path)
        try:
            age = float(ageStr)
        except ValueError:
            age = None
        if sex == "Undetermined":
            sex = None
        return age,sex

def getDir(dir, f_types=None):
    """Recursively search specified directory and return lists of full paths for all files of
    specified types.

    Adapted from GetDir() in eegreportparser.py

    Keyword arguments:
    dir -- the parent directory in which to search
    f_types -- List of strings indicating what file types to search for. "txt", "edf", or "set"

    Returns:
    files -- A dict of lists of file types, with key values as specified in f_types.
    """

    eligible_f_types = ["txt", "edf", "set"]

    if f_types is None:
        f_types = eligible_f_types
    elif type(f_types) is not list:
        raise TypeError("f_types should be a list e.g. f_types=['txt'] or f_types=['txt','edf']")

    # Make note of whether or not to collect each file type, and initialise files appropriately:
    fetch_type = {}
    files = {}
    for eft in eligible_f_types:
        if eft in f_types:
            fetch_type.update( {eft : True} )
            files.update( {eft : []} )
        else:
            fetch_type.update(False)

    # where TUE is the folder containing EEG data (txt and edf files) - with more data this will have to be more structured
    for file in os.scandir(dir):
        if file.is_dir():
            # Go deeper
            subd_files = getDir(file.path, f_types)
            # Append results from subdirectory(s)
            for ft,file_list in subd_files.items():
                files[ft] += file_list

        else:
            if fetch_type["txt"] and file.name.endswith(".txt"):
                files["txt"].append(file)
            elif fetch_type["edf"] and file.name.endswith(".edf"):
                files["edf"].append(file)
            elif fetch_type["set"] and file.name.endswith(".set"):
                files["set"].append(file)

    return files