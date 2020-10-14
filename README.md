ASCAM can be used to browse, organize and analyze episodic recordings of single ion channel currents.

## Installation
Requires python>=3.6 and Numpy>=1.16 
Further required packages can be found in 'requirements.txt' and installed with 
`pip install -r requirements.txt`.
Note: Unfortunately both ASCAM and one of its dependencies require Numpy, this
means that numpy needs to be installed separately before ASCAM can be installed.

A straightforward installation can be achieved by first installing Anaconda to get the necessary libraries. At the time of writing, a working version for Mac is https://repo.anaconda.com/archive/Anaconda3-5.3.0-MacOSX-x86_64.pkg 

After successful installation of Anaconda, if you have Git installed, you can clone the ASCAM directory from Github onto your machine with the following command in the Terminal: *git clone https://github.com/AGPlested/ASCAM*. But if you had Git installed, you almost certainly knew that already. 

20-03-01: Note, with the migration to Qt, some problems may be encountered on the Mac if you already have installations of Qt4+. Our investigations so far suggest a fresh install of Anaconda can help. 

If you don't have Git, you can just copy ASCAM directory and then navigate to it manually.

## Use
Start ASCAM from within its own directory by executing the file `run`. (Enter `./run` in the terminal)

Note: Tables in axograph and matlab have named columns ASCAM uses these names to determine what data is dealing with. Therefore the column containing the recorded current should contain either "current", "trace" or "Ipatch", the name of the column holding the recorded piezo voltage should contain the string "piezo" and the name of the command voltage column should contain "command".

There is an example raw data file of an AMPA receptor single channel patch in the ASCAM/data folder. This recording was sampled at 40 kHz.

![macOS Screenshot](cuteSCAM.png)
