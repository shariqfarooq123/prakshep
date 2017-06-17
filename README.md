# OCR | Prakshep
This is a django project written in python 2 with the aim to produce an ocr api

## Dependencies
This project uses modules from anaconda as well as pypi. Anaconda must be installed!
The corresponding dependencies are listed in `conda-requirements.txt` and `requirements.txt` respectively.

To avoid any dependency conflicts, create a conda virtual environment:
```
conda create --name ocr
source activate ocr
```
Now we are in the virtual environment, lets install dependencies.

To install pypi modules:

`pip install -r requirements.txt`

To install conda modules:

`conda install --yes --file conda-requirements.txt`


## Usage
For web interface, start a server using:
``` 
python manage.py startserver
```

The Command Line Interface (CLI) is provided by `prakshepocr/extract_cli.py`

```
cd prakshepocr
python extract_cli.py --help
```
Output:
```
usage: extract_cli.py [-h] [-a] [-v] filename

positional arguments:
  filename       filename of the image containing the card

optional arguments:
  -h, --help     show this help message and exit
  -a, --aadhaar  card is aadhaar (default is pan)
  -v, --verbose  enable logging

```

# Encryption Warning
The data used for testing and template matching is encrypted for privacy issues. Unlock the data folder by decrypting it
with the help of the encryption helper file named as `encryptor.py` present in `prakshepocr` folder

## Encryption helper usage 
```
cd prakshepocr
python encryptor.py --help
```
Output:
```
usage: encryptor.py [-h] {encrypt,decrypt} password

positional arguments:
  {encrypt,decrypt}  specify whether to encode or decode
  password           password used to encrypt/decrypt

optional arguments:
  -h, --help         show this help message and exit
```

### Decryption
Use following commands to decrypt. Let the password be `your_password`
```
python encryptor.py decrypt your_password
```
### Encryption
Similary, you can encrypt the data folder back with your own password before uploading the code to cloud

```
python encryptor.py encrypt your_password
```

