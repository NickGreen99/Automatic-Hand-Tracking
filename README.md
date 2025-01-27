# Automatic Hand Tracking

```shell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
```

If you are installing on Windows, it's strongly recommended to use Windows Subsystem for Linux (WSL) with Ubuntu to run the .sh file. Keep in mind for the sh file to work the line endings need to be `LF` not CRLF which is the default for Windows system. Therefore you need a text editor to change it to LF, save and then run the following commands. Another thing you can do is run the shell script in WSL and drag the .pt files in the checkpoints folder.

```shell
cd checkpoints
bash download_ckpts.sh
cd ..
```



